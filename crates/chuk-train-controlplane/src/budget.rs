//! Cost governance (spec §8): budget caps and the projected-spend check that
//! `provision` / `extend_lease` refuse on.
//!
//! Pure functions over (budgets, ledger, live leases) so the policy is
//! testable without a store: *spent* is realised `lease_end` cost inside the
//! budget's period, *committed* is the projected cost of live leases, and a
//! candidate action breaches when `spent + committed + candidate > cap`.

use chuk_train_proto::{
    Budget, Lease, LedgerEntry, SpendLine, SpendReport, UnixSeconds, BUDGET_PERIOD_ALL,
    BUDGET_PERIOD_MONTH, BUDGET_SCOPE_GLOBAL, BUDGET_SCOPE_PROVIDER_PREFIX,
    LEDGER_EVENT_LEASE_END,
};

const SECS_PER_DAY: i64 = 86_400;

/// A budget the candidate action would blow through.
#[derive(Debug, Clone, PartialEq)]
pub struct Breach {
    pub scope: String,
    pub period: String,
    pub cap: f64,
    /// spent + committed + candidate — what the cap would have to cover.
    pub projected: f64,
}

impl std::fmt::Display for Breach {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "budget breach: scope {} caps {:.2}/{} but this would project {:.2} \
             (raise the cap with set_budget, tear down idle workers, or wait for \
             the period to roll)",
            self.scope, self.cap, self.period, self.projected
        )
    }
}

pub fn validate_scope(scope: &str) -> Result<(), String> {
    if scope == BUDGET_SCOPE_GLOBAL {
        return Ok(());
    }
    match scope.strip_prefix(BUDGET_SCOPE_PROVIDER_PREFIX) {
        Some(name) if !name.is_empty() => Ok(()),
        _ => Err(format!(
            "unsupported budget scope {scope:?}: use {BUDGET_SCOPE_GLOBAL:?} or \
             \"{BUDGET_SCOPE_PROVIDER_PREFIX}<name>\" (label scopes are not enforced yet)"
        )),
    }
}

pub fn validate_period(period: &str) -> Result<(), String> {
    match period {
        BUDGET_PERIOD_MONTH | BUDGET_PERIOD_ALL => Ok(()),
        other => Err(format!(
            "unsupported budget period {other:?}: use {BUDGET_PERIOD_MONTH:?} or {BUDGET_PERIOD_ALL:?}"
        )),
    }
}

/// Unix time the period containing `now` started at, or `None` for all-time.
pub fn period_start(period: &str, now: UnixSeconds) -> Option<UnixSeconds> {
    match period {
        BUDGET_PERIOD_MONTH => Some(month_start(now)),
        _ => None,
    }
}

/// Start of the current UTC calendar month (civil-calendar math, no chrono
/// dependency; days↔date via Howard Hinnant's algorithms).
pub fn month_start(now: UnixSeconds) -> UnixSeconds {
    let days = (now as i64).div_euclid(SECS_PER_DAY);
    let (y, m, _) = civil_from_days(days);
    (days_from_civil(y, m, 1) * SECS_PER_DAY) as UnixSeconds
}

/// (year, month, day) for a count of days since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Days since 1970-01-01 for a civil (year, month, day).
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn budget_provider(scope: &str) -> Option<&str> {
    scope.strip_prefix(BUDGET_SCOPE_PROVIDER_PREFIX)
}

/// Realised spend on `provider` (all providers when `None`) since `since`.
fn spent(ledger: &[LedgerEntry], provider: Option<&str>, since: Option<UnixSeconds>) -> f64 {
    ledger
        .iter()
        .filter(|e| e.event == LEDGER_EVENT_LEASE_END)
        .filter(|e| since.is_none_or(|s| e.ts >= s))
        .filter(|e| provider.is_none_or(|p| e.provider == p))
        .map(|e| e.cost)
        .sum()
}

/// Projected cost of live leases on `provider` (all when `None`).
fn committed(live: &[Lease], provider: Option<&str>) -> f64 {
    live.iter()
        .filter(|l| provider.is_none_or(|p| l.provider == p))
        .map(Lease::projected_cost)
        .sum()
}

/// Would spending `candidate_cost` more on `provider` breach any applicable
/// budget? Checks `global` and `provider:<name>` scopes; first breach wins.
pub fn evaluate(
    budgets: &[Budget],
    ledger: &[LedgerEntry],
    live: &[Lease],
    provider: &str,
    candidate_cost: f64,
    now: UnixSeconds,
) -> Option<Breach> {
    for budget in budgets {
        let scope_provider = budget_provider(&budget.scope);
        let applies =
            budget.scope == BUDGET_SCOPE_GLOBAL || scope_provider == Some(provider);
        if !applies {
            continue;
        }
        let since = period_start(&budget.period, now);
        let projected = spent(ledger, scope_provider, since)
            + committed(live, scope_provider)
            + candidate_cost;
        if projected > budget.cap {
            return Some(Breach {
                scope: budget.scope.clone(),
                period: budget.period.clone(),
                cap: budget.cap,
                projected,
            });
        }
    }
    None
}

/// Worst-case dollar estimate for a submission (spec §8 pre-flight): the run's
/// wall-clock ceiling priced at the most expensive live lease. Zero when
/// nothing leased is billing — persistent/free workers have no hourly price.
pub fn estimate_run_cost(live: &[Lease], timeout_s: u64) -> f64 {
    let max_price_hr = live.iter().map(|l| l.price_hr).fold(0.0, f64::max);
    max_price_hr * timeout_s as f64 / 3600.0
}

/// The spend report (spec §6 `spend_status(period)`): per-provider committed +
/// period-spent, with cap/headroom attached where a matching-period budget
/// exists.
pub fn report(
    budgets: &[Budget],
    ledger: &[LedgerEntry],
    live: &[Lease],
    period: &str,
    now: UnixSeconds,
) -> SpendReport {
    use std::collections::BTreeSet;
    let since = period_start(period, now);
    let mut providers: BTreeSet<String> = BTreeSet::new();
    providers.extend(live.iter().map(|l| l.provider.clone()));
    providers.extend(
        ledger
            .iter()
            .filter(|e| e.event == LEDGER_EVENT_LEASE_END)
            .map(|e| e.provider.clone()),
    );
    // Budgeted providers show up even before any spend, so caps are visible.
    providers.extend(budgets.iter().filter_map(|b| budget_provider(&b.scope).map(str::to_owned)));

    let budget_for = |scope: &str| {
        budgets
            .iter()
            .find(|b| b.scope == scope && b.period == period)
    };
    let lines: Vec<SpendLine> = providers
        .into_iter()
        .map(|provider| {
            let line_spent = spent(ledger, Some(&provider), since);
            let line_committed = committed(live, Some(&provider));
            let cap = budget_for(&format!("{BUDGET_SCOPE_PROVIDER_PREFIX}{provider}"))
                .map(|b| b.cap);
            SpendLine {
                headroom: cap.map(|c| c - line_spent - line_committed),
                cap,
                committed: line_committed,
                spent: line_spent,
                provider,
            }
        })
        .collect();
    let total_committed = committed(live, None);
    let total_spent = spent(ledger, None, since);
    let global_cap = budget_for(BUDGET_SCOPE_GLOBAL).map(|b| b.cap);
    SpendReport {
        period: period.to_owned(),
        global_headroom: global_cap.map(|c| c - total_spent - total_committed),
        global_cap,
        total_committed,
        total_spent,
        lines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chuk_train_proto::{LeaseState, WorkerId};

    fn lease(provider: &str, price_hr: f64, granted_min: f64) -> Lease {
        Lease {
            worker_id: WorkerId(format!("{provider}-w")),
            provider: provider.into(),
            instance_id: "i".into(),
            price_hr,
            granted_min,
            drain_window_min: 5.0,
            started_at: 0.0,
            state: LeaseState::Active,
            extensions: Vec::new(),
        }
    }

    fn end_entry(provider: &str, ts: f64, cost: f64) -> LedgerEntry {
        LedgerEntry {
            ts,
            worker_id: WorkerId(format!("{provider}-w")),
            provider: provider.into(),
            event: LEDGER_EVENT_LEASE_END.into(),
            minutes: 60.0,
            cost,
        }
    }

    fn budget(scope: &str, cap: f64, period: &str) -> Budget {
        Budget {
            scope: scope.into(),
            cap,
            period: period.into(),
            updated_at: 0.0,
        }
    }

    // 2026-07-20 12:00:00 UTC and the month boundary around it.
    const MID_JULY: f64 = 1_784_548_800.0;

    #[test]
    fn month_start_is_the_first_of_the_month_utc() {
        let start = month_start(MID_JULY);
        let days = (start as i64) / SECS_PER_DAY;
        assert_eq!(civil_from_days(days), (2026, 7, 1));
        assert_eq!((start as i64) % SECS_PER_DAY, 0);
        // Idempotent: the month start is inside its own month.
        assert_eq!(month_start(start), start);
        // A tick before the boundary lands in the previous month.
        assert_eq!(civil_from_days(((start - 1.0) as i64).div_euclid(SECS_PER_DAY)).1, 6);
    }

    #[test]
    fn scope_and_period_validation() {
        assert!(validate_scope("global").is_ok());
        assert!(validate_scope("provider:vast").is_ok());
        assert!(validate_scope("provider:").is_err());
        assert!(validate_scope("label:cn7").is_err());
        assert!(validate_period("month").is_ok());
        assert!(validate_period("all").is_ok());
        assert!(validate_period("week").is_err());
    }

    #[test]
    fn provider_budget_counts_only_its_provider() {
        let budgets = vec![budget("provider:vast", 10.0, "all")];
        let ledger = vec![end_entry("vast", 1.0, 6.0), end_entry("colab", 1.0, 100.0)];
        // 6 spent + 3 candidate = 9 ≤ 10: fine.
        assert!(evaluate(&budgets, &ledger, &[], "vast", 3.0, MID_JULY).is_none());
        // 6 spent + 5 candidate = 11 > 10: breach.
        let breach = evaluate(&budgets, &ledger, &[], "vast", 5.0, MID_JULY).expect("breach");
        assert_eq!(breach.scope, "provider:vast");
        assert_eq!(breach.projected, 11.0);
        // Other providers are not constrained by vast's budget.
        assert!(evaluate(&budgets, &ledger, &[], "colab", 500.0, MID_JULY).is_none());
    }

    #[test]
    fn global_budget_counts_everything_and_live_leases_commit() {
        let budgets = vec![budget("global", 20.0, "all")];
        let ledger = vec![end_entry("vast", 1.0, 8.0)];
        let live = vec![lease("colab", 6.0, 60.0)]; // committed 6.0
        // 8 + 6 + 5 = 19 ≤ 20: fine.
        assert!(evaluate(&budgets, &ledger, &live, "vast", 5.0, MID_JULY).is_none());
        // 8 + 6 + 7 = 21 > 20: breach.
        assert!(evaluate(&budgets, &ledger, &live, "vast", 7.0, MID_JULY).is_some());
    }

    #[test]
    fn month_budget_ignores_last_months_spend() {
        let budgets = vec![budget("provider:vast", 10.0, "month")];
        let last_month = month_start(MID_JULY) - 1.0;
        let ledger = vec![end_entry("vast", last_month, 9.0), end_entry("vast", MID_JULY, 4.0)];
        // Only the in-month 4.0 counts: 4 + 5 = 9 ≤ 10.
        assert!(evaluate(&budgets, &ledger, &[], "vast", 5.0, MID_JULY).is_none());
        // An `all` budget would have counted both: 13 + 5 > 10.
        let all_time = vec![budget("provider:vast", 10.0, "all")];
        assert!(evaluate(&all_time, &ledger, &[], "vast", 5.0, MID_JULY).is_some());
    }

    #[test]
    fn estimate_prices_the_timeout_at_the_dearest_live_lease() {
        let live = vec![lease("vast", 2.0, 60.0), lease("lambda", 3.0, 60.0)];
        // 2h at $3/hr — the dearest lease is the worst case.
        assert_eq!(estimate_run_cost(&live, 2 * 3600), 6.0);
        // No billing workers → nothing to price the run against.
        assert_eq!(estimate_run_cost(&[], 12 * 3600), 0.0);
    }

    #[test]
    fn report_attaches_caps_and_headroom() {
        let budgets = vec![
            budget("global", 100.0, "month"),
            budget("provider:vast", 30.0, "month"),
            budget("provider:lambda", 50.0, "all"), // different period: not attached
        ];
        let ledger = vec![end_entry("vast", MID_JULY, 12.0)];
        let live = vec![lease("vast", 3.0, 120.0)]; // committed 6.0
        let report = report(&budgets, &ledger, &live, "month", MID_JULY);
        assert_eq!(report.period, "month");
        assert_eq!(report.global_cap, Some(100.0));
        assert_eq!(report.global_headroom, Some(100.0 - 12.0 - 6.0));
        let vast = report.lines.iter().find(|l| l.provider == "vast").expect("vast");
        assert_eq!(vast.cap, Some(30.0));
        assert_eq!(vast.headroom, Some(30.0 - 12.0 - 6.0));
        // The budgeted-but-unspent provider still appears, with full headroom —
        // except its budget period differs, so no cap is attached.
        let lambda = report.lines.iter().find(|l| l.provider == "lambda").expect("lambda");
        assert_eq!(lambda.cap, None);
        assert_eq!(lambda.spent, 0.0);
    }
}
