//! A small development server: the real world runtime behind the UI, in a
//! browser, without Tauri.
//!
//!     cargo run --release -p hand-runtime --example serve [port] [ui_dir]
//!
//! It serves `ui/` and answers `POST /invoke/<command>` with the same JSON the
//! Tauri commands return. A shim script is added to `index.html` so that
//! `window.__TAURI__.core.invoke` exists: the eleven world commands go to this
//! server, and the older hand/vision commands fall back to the UI's browser
//! mock (they need the Tauri build to be real).
//!
//! It listens on 127.0.0.1 only and has no authentication. It is for looking at
//! the simulator on your own machine, not for serving anyone else.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;

use hand_runtime::{Runtime, RuntimeConfig};
use serde_json::Value;

const SHIM: &str = r#"<script>
// 개발 서버 shim: 월드 명령은 이 서버(실제 Rust 런타임)로, 나머지는 브라우저 모의 백엔드로 보낸다.
(() => {
  const WORLD = new Set(['world_new','sim_step','world_frame','command_submit','quest_list','perception_scan','body_status','sds_status','ts_forecast','log_tail','models_status',
    'test_start','test_step','test_status','test_keyframe','test_video','test_stats','test_cancel','body_presets','body_plan_eval','body_plan_set','world_body','test_db_mirror',
    'inventory_drop','world_reset','body_reset','world_terrain','nn_status','physics_status']);
  let mock = null;
  window.__TAURI__ = { core: { invoke: async (cmd, args) => {
    if (!WORLD.has(cmd)) {
      mock = mock || (await import('/js/mock-backend.js'));
      return mock.invoke(cmd, args);
    }
    const r = await fetch('/invoke/' + cmd, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(args ?? {}) });
    const j = await r.json();
    if (!r.ok) throw (j && j.error) || ('HTTP ' + r.status);
    return j;
  } } };
  window.__devServer = true;
})();
</script>"#;

fn content_type(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "webp" => "image/webp",
        "woff2" => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn respond(s: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) {
    let head = format!("HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n", body.len());
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(body);
}

/// `/js/main.js` to `<ui>/js/main.js`; anything that climbs out of `ui` is refused.
fn safe_path(ui: &Path, url_path: &str) -> Option<PathBuf> {
    let rel = url_path.split('?').next().unwrap_or("").trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    let mut out = ui.to_path_buf();
    for c in Path::new(rel).components() {
        match c {
            Component::Normal(p) => out.push(p),
            _ => return None,
        }
    }
    Some(out)
}

fn handle(mut s: TcpStream, ui: &Path, rt: &Mutex<Runtime>) {
    let mut reader = BufReader::new(match s.try_clone() {
        Ok(c) => c,
        Err(_) => return,
    });
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("/").to_string());
    let mut headers: HashMap<String, String> = HashMap::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let len: usize = headers.get("content-length").and_then(|v| v.parse().ok()).unwrap_or(0).min(8 << 20);
    let mut body = vec![0u8; len];
    if len > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }

    if method == "POST" {
        if let Some(cmd) = path.strip_prefix("/invoke/") {
            let args: Value = if body.is_empty() { Value::Null } else { serde_json::from_slice(&body).unwrap_or(Value::Null) };
            let result = if cmd == "test_db_mirror" {
                mirror(rt)
            } else {
                match rt.lock() {
                    Ok(mut r) => r.api_call(cmd, &args),
                    Err(_) => Err("월드 잠금이 깨졌습니다".to_string()),
                }
            };
            match result {
                Ok(v) => respond(&mut s, "200 OK", "application/json", v.to_string().as_bytes()),
                Err(e) => respond(&mut s, "400 Bad Request", "application/json", serde_json::json!({ "error": e }).to_string().as_bytes()),
            }
            return;
        }
    }
    if method != "GET" {
        respond(&mut s, "405 Method Not Allowed", "text/plain", b"method not allowed");
        return;
    }
    match safe_path(ui, &path).and_then(|p| std::fs::read(&p).ok().map(|b| (p, b))) {
        Some((p, bytes)) if p.file_name().is_some_and(|n| n == "index.html") => {
            let html = String::from_utf8_lossy(&bytes).replacen("</head>", &format!("{SHIM}\n</head>"), 1);
            respond(&mut s, "200 OK", content_type(&p), html.as_bytes());
        }
        Some((p, bytes)) => respond(&mut s, "200 OK", content_type(&p), &bytes),
        None => respond(&mut s, "404 Not Found", "text/plain", b"not found"),
    }
}

/// The test records written since the last call, into the Lance store
/// (`LANCE_DIR`, default `data/lance`), as the Tauri app does.
fn mirror(rt: &Mutex<Runtime>) -> Result<Value, String> {
    let (batch, stats) = {
        let mut r = rt.lock().map_err(|_| "월드 잠금이 깨졌습니다".to_string())?;
        let db = r.test_db_mut();
        (db.take_mirror(), db.stats())
    };
    let dir = std::env::var("LANCE_DIR").map(PathBuf::from).unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/lance"));
    let uri = dir.to_str().ok_or("저장 경로가 UTF-8 이 아닙니다")?.to_string();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().map_err(|e| e.to_string())?;
    let counts = rt
        .block_on(async {
            let store = hand_data::LanceStore::open(&uri).await?;
            hand_data::test_tables::write_mirror(&store, &batch, &stats).await
        })
        .map_err(|e| format!("{e:#}"))?;
    Ok(serde_json::json!({"store_dir": dir.display().to_string(), "written": counts}))
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let port: u16 = args.next().and_then(|p| p.parse().ok()).unwrap_or(8787);
    let ui = args.next().map(PathBuf::from).unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ui"));
    let lexicon = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/world_lexicon.json"))?;
    let data_dir = std::env::var("RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/runtime"));
    let rt = Arc::new(Mutex::new(Runtime::new(RuntimeConfig::default(), &lexicon, Some(&data_dir))?));
    let ui = Arc::new(ui.canonicalize()?);
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    println!("http://127.0.0.1:{port}/  (ui: {}, 로그·SDS: {})", ui.display(), data_dir.display());
    for conn in listener.incoming().flatten() {
        let (rt, ui) = (Arc::clone(&rt), Arc::clone(&ui));
        thread::spawn(move || handle(conn, &ui, &rt));
    }
    Ok(())
}
