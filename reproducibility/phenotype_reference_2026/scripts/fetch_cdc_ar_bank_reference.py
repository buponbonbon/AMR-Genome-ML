#!/usr/bin/env python3
"""Download public CDC panel pages and their MIC/gene data export, without ordering isolates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import http.cookiejar
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


class PanelParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hidden = {}
        self.rows = []
        self.row = None
        self.cell = None
        self.table = False
        self.isolates = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("type") == "hidden":
            self.hidden[attrs.get("name", "")] = attrs.get("value", "")
        if tag == "table" and attrs.get("id") == "dataTable":
            self.table = True
        if self.table and tag == "tr":
            self.row = []
        if self.table and tag in {"td", "th"}:
            self.cell = []
        if self.table and tag == "a" and "IsolateDetail" in attrs.get("href", ""):
            self.isolates.append(attrs["href"])

    def handle_data(self, data):
        self.text.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if self.table and tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        if self.table and tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        if tag == "table":
            self.table = False


def read_response(opener, request, path):
    with opener.open(request, timeout=45) as response:
        content = response.read(30 * 1024 * 1024 + 1)
        if len(content) > 30 * 1024 * 1024:
            raise ValueError("Public export exceeded 30 MiB guard")
        path.write_bytes(content)
        return {"file": path.name, "http_status": response.status, "final_url": response.url,
                "bytes": len(content), "content_type": response.headers.get("Content-Type"),
                "content_disposition": response.headers.get("Content-Disposition"),
                "sha256": hashlib.sha256(content).hexdigest(),
                "retrieved_utc": datetime.now(timezone.utc).isoformat()}


def fetch(panel, out):
    panel_id = int(panel["id"])
    url = f"https://wwwn.cdc.gov/ARIsolateBank/Panel/PanelDetail?ID={panel_id}"
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = [("User-Agent", "AMRReferenceAudit/1.0")]
    result = {"panel": panel, "url": url}
    try:
        page = out / f"panel_{panel_id}.html"
        result["page"] = read_response(opener, url, page)
        parser = PanelParser()
        parser.feed(page.read_text())
        result["panel_table_rows"] = parser.rows
        result["isolate_detail_urls"] = [urllib.parse.urljoin(url, x) for x in parser.isolates]
        result["linked_biosamples"] = sorted(set(re.findall(r"SAM[NED][A-Z]?\d+", page.read_text())))
        result["BMD_stated_on_panel_page"] = "broth microdilution" in " ".join(parser.text).lower()
        button = "ctl00$MainContent$PanelInfoDownload"
        if button not in page.read_text():
            raise ValueError("Expected public MIC download control absent")
        # This specific POST only requests the public downloadable data file.
        # No order/cart/account controls are included.
        fields = {k: v for k, v in parser.hidden.items() if k}
        fields["ctl00$MainContent$ddlPanelListing"] = ""
        fields[button] = "Download MIC & gene data"
        request = urllib.request.Request(url, data=urllib.parse.urlencode(fields).encode(),
                                         headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": url})
        result["export"] = read_response(opener, request, out / f"panel_{panel_id}_MIC_gene.export")
        kind = (result["export"]["content_type"] or "").lower()
        result["export_download_status"] = "RECEIVED_REQUIRES_SCHEMA_INSPECTION" if "html" not in kind else "HTML_RESPONSE_NOT_CONFIRMED_EXPORT"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda panel: fetch(panel, args.out), protocol["panels"]))
    report = {"protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(), "results": results,
              "limitations": ["Linked genomes and MIC exports require schema and overlap verification.",
                              "Multiple CDC panels can contain the same isolate; panels are not independent cohorts.",
                              "No publication, ordering or author contact was performed."]}
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print(json.dumps({"panel_id": result["panel"]["id"], "biosamples": len(result.get("linked_biosamples", [])),
                          "export_status": result.get("export_download_status"), "error": result.get("error")}))


if __name__ == "__main__":
    main()
