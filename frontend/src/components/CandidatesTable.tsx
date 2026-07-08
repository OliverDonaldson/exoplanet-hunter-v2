import { useEffect, useMemo, useState } from "react";
import { candidatesCsvUrl, fetchCandidates } from "../api/client";
import type { CandidateRow, CandidatesPage } from "../api/types";

const PAGE_SIZE = 50;

const COLUMNS: { key: string; label: string; sortable: boolean; render: (r: CandidateRow) => string }[] = [
  { key: "name", label: "Candidate", sortable: true, render: (r) => r.name },
  { key: "tic_id", label: "TIC ID", sortable: true, render: (r) => String(r.tic_id) },
  { key: "source", label: "Source", sortable: false, render: (r) => r.source },
  { key: "disposition", label: "TFOPWG", sortable: true, render: (r) => r.disposition ?? "—" },
  { key: "tess_mag", label: "Tmag", sortable: true, render: (r) => fmt(r.tess_mag, 2) },
  { key: "period_days", label: "Period (d)", sortable: true, render: (r) => fmt(r.period_days, 3) },
  { key: "duration_hours", label: "Dur (h)", sortable: true, render: (r) => fmt(r.duration_hours, 2) },
  { key: "depth_ppm", label: "Depth (ppm)", sortable: true, render: (r) => fmt(r.depth_ppm, 0) },
  { key: "planet_radius_re", label: "R (R⊕)", sortable: true, render: (r) => fmt(r.planet_radius_re, 2) },
  { key: "stellar_teff_k", label: "Teff (K)", sortable: true, render: (r) => fmt(r.stellar_teff_k, 0) },
];

function fmt(value: number | null, digits: number): string {
  return value === null ? "—" : value.toFixed(digits);
}

export default function CandidatesTable() {
  const [search, setSearch] = useState("");
  const [disposition, setDisposition] = useState("");
  const [source, setSource] = useState("");
  const [sortBy, setSortBy] = useState("tess_mag");
  const [order, setOrder] = useState<"asc" | "desc">("asc");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<CandidatesPage | null>(null);
  const [error, setError] = useState<string | null>(null);

  const query = useMemo(
    () => ({ search, disposition, source, sort_by: sortBy, order, limit: PAGE_SIZE, offset }),
    [search, disposition, source, sortBy, order, offset],
  );

  useEffect(() => {
    // Debounce so typing in the search box doesn't fire a request per keystroke.
    const t = setTimeout(() => {
      fetchCandidates(query)
        .then((p) => {
          setPage(p);
          setError(null);
        })
        .catch((e: Error) => setError(e.message));
    }, 250);
    return () => clearTimeout(t);
  }, [query]);

  function toggleSort(key: string) {
    if (sortBy === key) {
      setOrder(order === "asc" ? "desc" : "asc");
    } else {
      setSortBy(key);
      setOrder("asc");
    }
    setOffset(0);
  }

  const total = page?.total ?? 0;
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + PAGE_SIZE, total);

  return (
    <section>
      <h2>Candidate catalogue</h2>
      <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.75rem" }}>
        <input
          placeholder="Search name / TIC / comments…"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setOffset(0);
          }}
          style={{ flex: "1 1 16rem", padding: "0.4rem" }}
        />
        <select value={disposition} onChange={(e) => { setDisposition(e.target.value); setOffset(0); }}>
          <option value="">All dispositions</option>
          {["PC", "CP", "KP", "APC", "FP", "FA"].map((d) => (
            <option key={d} value={d}>{d}</option>
          ))}
          <option value="none">(none)</option>
        </select>
        <select value={source} onChange={(e) => { setSource(e.target.value); setOffset(0); }}>
          <option value="">TOI + CTOI</option>
          <option value="TOI">TOI only</option>
          <option value="CTOI">CTOI only</option>
        </select>
        <a href={candidatesCsvUrl(query)} download="candidates.csv">
          <button type="button">Download CSV ({total.toLocaleString()} rows)</button>
        </a>
      </div>

      {error && <p style={{ color: "#b91c1c" }}>Failed to load catalogue: {error}</p>}

      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "0.85rem" }}>
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  onClick={col.sortable ? () => toggleSort(col.key) : undefined}
                  style={{
                    textAlign: "left",
                    padding: "0.4rem 0.6rem",
                    borderBottom: "2px solid #999",
                    cursor: col.sortable ? "pointer" : "default",
                    whiteSpace: "nowrap",
                  }}
                >
                  {col.label}
                  {sortBy === col.key ? (order === "asc" ? " ↑" : " ↓") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {page?.rows.map((row) => (
              <tr key={`${row.source}-${row.name}`}>
                {COLUMNS.map((col) => (
                  <td
                    key={col.key}
                    style={{ padding: "0.35rem 0.6rem", borderBottom: "1px solid #ddd", whiteSpace: "nowrap" }}
                  >
                    {col.render(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginTop: "0.75rem" }}>
        <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
          ← Prev
        </button>
        <span>
          {from.toLocaleString()}–{to.toLocaleString()} of {total.toLocaleString()}
        </span>
        <button type="button" disabled={to >= total} onClick={() => setOffset(offset + PAGE_SIZE)}>
          Next →
        </button>
      </div>
    </section>
  );
}
