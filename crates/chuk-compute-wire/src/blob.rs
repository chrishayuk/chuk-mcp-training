//! The worker↔control-plane blob-transfer contract: a worker asks the control
//! plane where to move a blob, and the response says where. Presign-first, so
//! large artifacts bypass the control plane whenever the backend can sign a URL.

use serde::{Deserialize, Serialize};

/// Which direction a worker wants to transfer a blob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlobMethod {
    Put,
    Get,
}

/// A worker asking the control plane where to transfer a blob (spec §12).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlobUrlRequest {
    pub key: String,
    pub method: BlobMethod,
}

/// Where to transfer it. With an S3/R2 backend `url` is a presigned URL the
/// worker hits directly (bytes bypass the control plane); with the filesystem
/// backend it points back at the control plane's own upload/fetch endpoint and
/// `requires_grant_header` is set so the worker attaches its grant token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlobUrlResponse {
    pub url: String,
    #[serde(default)]
    pub requires_grant_header: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_method_is_snake_case() {
        assert_eq!(serde_json::to_string(&BlobMethod::Put).unwrap(), r#""put""#);
        assert_eq!(serde_json::to_string(&BlobMethod::Get).unwrap(), r#""get""#);
        assert_eq!(
            serde_json::from_str::<BlobMethod>(r#""put""#).unwrap(),
            BlobMethod::Put
        );
        assert_eq!(
            serde_json::from_str::<BlobMethod>(r#""get""#).unwrap(),
            BlobMethod::Get
        );
    }

    #[test]
    fn blob_url_request_round_trips() {
        let req = BlobUrlRequest {
            key: "ckpt-hot/r1/step_5/model.safetensors".into(),
            method: BlobMethod::Put,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""method":"put""#), "{json}");
        assert_eq!(serde_json::from_str::<BlobUrlRequest>(&json).unwrap(), req);
    }

    #[test]
    fn blob_url_response_round_trips_and_defaults_grant_header() {
        // Presigned URL: the header flag is omitted → defaults to false.
        let signed: BlobUrlResponse =
            serde_json::from_str(r#"{"url":"https://r2.example/put?sig=abc"}"#).unwrap();
        assert_eq!(signed.url, "https://r2.example/put?sig=abc");
        assert!(!signed.requires_grant_header);

        // Filesystem fallback: the flag is set so the worker attaches its grant.
        let fallback = BlobUrlResponse {
            url: "http://cp/api/upload/key".into(),
            requires_grant_header: true,
        };
        let round: BlobUrlResponse =
            serde_json::from_str(&serde_json::to_string(&fallback).unwrap()).unwrap();
        assert_eq!(round, fallback);
    }
}
