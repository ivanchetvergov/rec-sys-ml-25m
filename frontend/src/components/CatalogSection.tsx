"use client";

import { MovieCard } from "@/components/MovieCard";
import type { Movie } from "@/lib/api";
import { fetchAllMovies } from "@/lib/api";
import { useEffect, useMemo, useState } from "react";

interface Props {
    initialMovies?: Movie[];
    onSelect?: (movie: Movie) => void;
}

type SortKey = "popularity" | "rating" | "year_desc" | "year_asc" | "reviews";

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
    { value: "popularity", label: "Popularity" },
    { value: "rating", label: "Rating" },
    { value: "reviews", label: "Most reviewed" },
    { value: "year_desc", label: "Newest first" },
    { value: "year_asc", label: "Oldest first" },
];

const MIN_REVIEWS_OPTIONS = [
    { label: "Any", value: 0 },
    { label: "100+", value: 100 },
    { label: "1k+", value: 1_000 },
    { label: "10k+", value: 10_000 },
    { label: "50k+", value: 50_000 },
];

function pill(active: boolean) {
    return {
        borderColor: active ? "var(--netflix-red)" : "rgba(255,255,255,0.15)",
        color: active ? "var(--netflix-red)" : "#a1a1aa",
        background: active ? "rgba(229,9,20,0.10)" : "transparent",
    } as React.CSSProperties;
}

function FilterChip({ label, onRemove }: { label: string; onRemove: () => void }) {
    return (
        <span
            className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full font-medium"
            style={{ background: "rgba(229,9,20,0.15)", color: "#f87171", border: "1px solid rgba(229,9,20,0.3)" }}
        >
            {label}
            <button onClick={onRemove} className="hover:text-white transition-colors leading-none" aria-label={`Remove filter ${label}`}>
                ✕
            </button>
        </span>
    );
}

function RangeInput({
    label,
    value,
    min,
    max,
    step = 1,
    format,
    onChange,
}: {
    label: string;
    value: number;
    min: number;
    max: number;
    step?: number;
    format?: (v: number) => string;
    onChange: (v: number) => void;
}) {
    const pct = ((value - min) / (max - min)) * 100;
    return (
        <div>
            <div className="flex justify-between mb-1.5">
                <span className="text-xs text-zinc-400">{label}</span>
                <span className="text-xs font-semibold text-white">
                    {format ? format(value) : value}
                </span>
            </div>
            <div className="relative h-5 flex items-center">
                <div className="w-full h-1 rounded-full" style={{ background: "rgba(255,255,255,0.12)" }}>
                    <div
                        className="h-1 rounded-full"
                        style={{ width: `${pct}%`, background: "var(--netflix-red)" }}
                    />
                </div>
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(e) => onChange(Number(e.target.value))}
                    className="absolute inset-0 w-full opacity-0 cursor-pointer h-5"
                />
                {/* Thumb indicator */}
                <div
                    className="absolute w-3.5 h-3.5 rounded-full pointer-events-none"
                    style={{
                        left: `calc(${pct}% - 7px)`,
                        background: "var(--netflix-red)",
                        border: "2px solid #fff",
                        boxShadow: "0 0 0 3px rgba(229,9,20,0.3)",
                    }}
                />
            </div>
        </div>
    );
}

export function CatalogSection({ initialMovies = [], onSelect }: Props) {
    const [query, setQuery] = useState("");
    const [selectedGenres, setSelectedGenres] = useState<Set<string>>(new Set());
    const [yearMin, setYearMin] = useState<number | null>(null);
    const [yearMax, setYearMax] = useState<number | null>(null);
    const [minRating, setMinRating] = useState(0);
    const [minReviews, setMinReviews] = useState(0);
    const [sortBy, setSortBy] = useState<SortKey>("popularity");
    const [filtersOpen, setFiltersOpen] = useState(false);

    // Full catalog — starts with server-provided slice, replaced by full fetch on mount
    const [allMovies, setAllMovies] = useState<Movie[]>(initialMovies);
    const [catalogLoading, setCatalogLoading] = useState(true);

    useEffect(() => {
        setCatalogLoading(true);
        fetchAllMovies()
            .then(setAllMovies)
            .catch(() => { /* keep initialMovies on error */ })
            .finally(() => setCatalogLoading(false));
    }, []);

    const movies = allMovies;

    // Derived ranges from dataset
    const { genres, yearRange } = useMemo(() => {
        const genreSet = new Set<string>();
        let yMin = Infinity;
        let yMax = -Infinity;
        for (const m of movies) {
            m.genres?.split("|").forEach((g) => genreSet.add(g));
            if (m.year) {
                yMin = Math.min(yMin, m.year);
                yMax = Math.max(yMax, m.year);
            }
        }
        return {
            genres: Array.from(genreSet).sort(),
            yearRange: { min: yMin === Infinity ? 1900 : yMin, max: yMax === -Infinity ? 2025 : yMax },
        };
    }, [movies]);

    const effectiveYearMin = yearMin ?? yearRange.min;
    const effectiveYearMax = yearMax ?? yearRange.max;

    // Filter + sort
    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        let result = movies.filter((m) => {
            if (q && !m.title.toLowerCase().includes(q)) return false;
            if (selectedGenres.size > 0) {
                const mg = new Set(m.genres?.split("|") ?? []);
                // All selected genres must be present (AND logic)
                for (const g of Array.from(selectedGenres)) if (!mg.has(g)) return false;
            }
            if (m.year && (m.year < effectiveYearMin || m.year > effectiveYearMax)) return false;
            if (minRating > 0 && (m.avg_rating == null || m.avg_rating < minRating)) return false;
            if (minReviews > 0 && (m.num_ratings == null || m.num_ratings < minReviews)) return false;
            return true;
        });

        result = [...result].sort((a, b) => {
            switch (sortBy) {
                case "rating": return (b.avg_rating ?? 0) - (a.avg_rating ?? 0);
                case "reviews": return (b.num_ratings ?? 0) - (a.num_ratings ?? 0);
                case "year_desc": return (b.year ?? 0) - (a.year ?? 0);
                case "year_asc": return (a.year ?? 0) - (b.year ?? 0);
                default: return (b.popularity_score ?? 0) - (a.popularity_score ?? 0);
            }
        });

        return result;
    }, [movies, query, selectedGenres, effectiveYearMin, effectiveYearMax, minRating, minReviews, sortBy]);

    const toggleGenre = (g: string) => {
        setSelectedGenres((prev) => {
            const next = new Set(prev);
            next.has(g) ? next.delete(g) : next.add(g);
            return next;
        });
    };

    const anyFilterActive =
        query.trim() !== "" ||
        selectedGenres.size > 0 ||
        yearMin !== null ||
        yearMax !== null ||
        minRating > 0 ||
        minReviews > 0 ||
        sortBy !== "popularity";

    const clearAll = () => {
        setQuery("");
        setSelectedGenres(new Set());
        setYearMin(null);
        setYearMax(null);
        setMinRating(0);
        setMinReviews(0);
        setSortBy("popularity");
    };

    // Active chips
    const activeChips: { label: string; onRemove: () => void }[] = [];
    if (query.trim()) activeChips.push({ label: `"${query.trim()}"`, onRemove: () => setQuery("") });
    for (const g of Array.from(selectedGenres)) activeChips.push({ label: g, onRemove: () => toggleGenre(g) });
    if (yearMin !== null || yearMax !== null)
        activeChips.push({ label: `${effectiveYearMin}–${effectiveYearMax}`, onRemove: () => { setYearMin(null); setYearMax(null); } });
    if (minRating > 0)
        activeChips.push({ label: `★ ≥ ${minRating.toFixed(1)}`, onRemove: () => setMinRating(0) });
    if (minReviews > 0)
        activeChips.push({ label: `${(minReviews >= 1000 ? (minReviews / 1000).toFixed(0) + "k" : minReviews)}+ reviews`, onRemove: () => setMinReviews(0) });
    if (sortBy !== "popularity")
        activeChips.push({ label: `Sort: ${SORT_OPTIONS.find((o) => o.value === sortBy)?.label}`, onRemove: () => setSortBy("popularity") });

    return (
        <section id="catalog" className="pb-20">
            {/* ── Header ─────────────────────────────────────────────── */}
            <div className="flex items-center justify-between mb-5">
                <div>
                    <h2 className="text-xl font-bold text-white">Browse Catalog</h2>
                    <p className="text-sm text-zinc-500 mt-0.5">
                        {catalogLoading
                            ? "Loading full catalog…"
                            : `${filtered.length.toLocaleString()} of ${movies.length.toLocaleString()} movies`}
                    </p>
                </div>
                {anyFilterActive && (
                    <button
                        onClick={clearAll}
                        className="text-xs text-zinc-500 hover:text-white transition-colors underline underline-offset-2"
                    >
                        Clear all filters
                    </button>
                )}
            </div>

            {/* ── Search + Filter toggle row ──────────────────────────── */}
            <div className="flex gap-3 mb-4">
                <div className="relative flex-1 max-w-md">
                    <svg xmlns="http://www.w3.org/2000/svg"
                        className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500"
                        fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
                    </svg>
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Search by title..."
                        className="w-full pl-9 pr-10 py-2.5 rounded-lg text-sm text-zinc-200 outline-none focus:ring-1"
                        style={{ background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.12)" }}
                    />
                    {query && (
                        <button onClick={() => setQuery("")}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-white transition-colors"
                            aria-label="Clear">✕</button>
                    )}
                </div>

                {/* Filters toggle */}
                <button
                    onClick={() => setFiltersOpen((v) => !v)}
                    className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors"
                    style={{
                        background: filtersOpen ? "rgba(229,9,20,0.15)" : "rgba(255,255,255,0.07)",
                        border: `1px solid ${filtersOpen ? "rgba(229,9,20,0.4)" : "rgba(255,255,255,0.12)"}`,
                        color: filtersOpen ? "#f87171" : "#a1a1aa",
                    }}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3 4h18M6 8h12M9 12h6M11 16h2" />
                    </svg>
                    Filters
                    {anyFilterActive && (
                        <span className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--netflix-red)" }} />
                    )}
                </button>

                {/* Sort */}
                <div className="relative">
                    <select
                        value={sortBy}
                        onChange={(e) => setSortBy(e.target.value as SortKey)}
                        className="appearance-none px-4 pr-8 py-2.5 rounded-lg text-sm font-medium outline-none cursor-pointer"
                        style={{
                            background: "rgba(255,255,255,0.07)",
                            border: "1px solid rgba(255,255,255,0.12)",
                            color: sortBy !== "popularity" ? "var(--netflix-red)" : "#a1a1aa",
                        }}
                    >
                        {SORT_OPTIONS.map((o) => (
                            <option key={o.value} value={o.value} style={{ background: "#1a1a1a" }}>
                                {o.label}
                            </option>
                        ))}
                    </select>
                    <svg xmlns="http://www.w3.org/2000/svg" className="absolute right-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 pointer-events-none text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                    </svg>
                </div>
            </div>

            {/* ── Expanded filter panel ───────────────────────────────── */}
            {filtersOpen && (
                <div
                    className="mb-6 rounded-xl p-5 flex flex-col gap-6"
                    style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)" }}
                >
                    {/* ── Year range ── */}
                    <div>
                        <p className="text-xs uppercase tracking-wider text-zinc-500 mb-3 font-semibold">Year</p>
                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="text-xs text-zinc-500 mb-1 block">From</label>
                                <input
                                    type="number"
                                    value={effectiveYearMin}
                                    min={yearRange.min}
                                    max={effectiveYearMax}
                                    onChange={(e) => setYearMin(Number(e.target.value) || null)}
                                    className="w-full px-3 py-2 rounded-lg text-sm text-zinc-200 outline-none"
                                    style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.10)" }}
                                />
                            </div>
                            <div>
                                <label className="text-xs text-zinc-500 mb-1 block">To</label>
                                <input
                                    type="number"
                                    value={effectiveYearMax}
                                    min={effectiveYearMin}
                                    max={yearRange.max}
                                    onChange={(e) => setYearMax(Number(e.target.value) || null)}
                                    className="w-full px-3 py-2 rounded-lg text-sm text-zinc-200 outline-none"
                                    style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.10)" }}
                                />
                            </div>
                        </div>
                    </div>

                    {/* ── Min rating ── */}
                    <div>
                        <p className="text-xs uppercase tracking-wider text-zinc-500 mb-3 font-semibold">Minimum rating</p>
                        <RangeInput
                            label="Avg rating"
                            value={minRating}
                            min={0}
                            max={5}
                            step={0.1}
                            format={(v) => v === 0 ? "Any" : `★ ${v.toFixed(1)}`}
                            onChange={setMinRating}
                        />
                    </div>

                    {/* ── Min reviews ── */}
                    <div>
                        <p className="text-xs uppercase tracking-wider text-zinc-500 mb-3 font-semibold">Minimum reviews</p>
                        <div className="flex flex-wrap gap-2">
                            {MIN_REVIEWS_OPTIONS.map((opt) => (
                                <button
                                    key={opt.value}
                                    onClick={() => setMinReviews(opt.value)}
                                    className="text-sm px-3 py-1.5 rounded-full border transition-colors"
                                    style={pill(minReviews === opt.value)}
                                >
                                    {opt.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* ── Genres ── */}
                    <div>
                        <div className="flex items-center justify-between mb-3">
                            <p className="text-xs uppercase tracking-wider text-zinc-500 font-semibold">
                                Genres
                                {selectedGenres.size > 0 && (
                                    <span className="ml-2 normal-case font-normal text-zinc-400">
                                        ({selectedGenres.size} selected — all must match)
                                    </span>
                                )}
                            </p>
                            {selectedGenres.size > 0 && (
                                <button
                                    onClick={() => setSelectedGenres(new Set())}
                                    className="text-xs text-zinc-500 hover:text-white transition-colors"
                                >
                                    Clear genres
                                </button>
                            )}
                        </div>
                        <div className="flex flex-wrap gap-2">
                            {genres.map((g) => (
                                <button
                                    key={g}
                                    onClick={() => toggleGenre(g)}
                                    className="text-sm px-3 py-1.5 rounded-full border transition-colors"
                                    style={pill(selectedGenres.has(g))}
                                >
                                    {g}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* ── Active filter chips ─────────────────────────────────── */}
            {activeChips.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-5">
                    {activeChips.map((chip) => (
                        <FilterChip key={chip.label} label={chip.label} onRemove={chip.onRemove} />
                    ))}
                </div>
            )}

            {/* ── Grid ───────────────────────────────────────────────── */}
            {catalogLoading && movies.length === 0 ? (
                <div className="h-64 flex items-center justify-center text-zinc-500 text-sm">
                    Loading catalog…
                </div>
            ) : filtered.length === 0 ? (
                <div className="py-20 text-center">
                    <p className="text-zinc-400 text-base font-semibold mb-2">No movies match your filters</p>
                    <button onClick={clearAll} className="text-sm text-zinc-500 hover:text-white transition-colors underline underline-offset-2">
                        Clear all filters
                    </button>
                </div>
            ) : (
                <div
                    className="grid gap-4"
                    style={{ gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))" }}
                >
                    {filtered.map((movie, i) => (
                        <MovieCard key={movie.id} movie={movie} rank={i + 1} onSelect={onSelect} />
                    ))}
                </div>
            )}
        </section>
    );
}
