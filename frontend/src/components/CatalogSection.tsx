"use client";

import { MovieCard } from "@/components/MovieCard";
import type { Movie } from "@/lib/api";
import { useMemo, useState } from "react";

interface Props {
    movies: Movie[];
    onSelect?: (movie: Movie) => void;
}

export function CatalogSection({ movies, onSelect }: Props) {
    const [query, setQuery] = useState("");
    const [activeGenre, setActiveGenre] = useState<string | null>(null);

    // Collect unique genres from all movies
    const genres = useMemo(() => {
        const set = new Set<string>();
        for (const m of movies) {
            m.genres?.split("|").forEach((g) => set.add(g));
        }
        return Array.from(set).sort();
    }, [movies]);

    // Filter
    const filtered = useMemo(() => {
        const q = query.trim().toLowerCase();
        return movies.filter((m) => {
            const matchesQuery = !q || m.title.toLowerCase().includes(q);
            const matchesGenre = !activeGenre || m.genres?.split("|").includes(activeGenre);
            return matchesQuery && matchesGenre;
        });
    }, [movies, query, activeGenre]);

    return (
        <section id="catalog" className="pb-20">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-xl font-bold text-white">Browse Catalog</h2>
                    <p className="text-sm text-zinc-500 mt-0.5">
                        {filtered.length} of {movies.length} movies
                    </p>
                </div>
            </div>

            {/* Search bar */}
            <div className="relative mb-5 max-w-md">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500"
                    fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
                >
                    <path strokeLinecap="round" strokeLinejoin="round"
                        d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
                </svg>
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search by title..."
                    className="w-full pl-9 pr-10 py-2.5 rounded-lg text-sm text-zinc-200 outline-none"
                    style={{
                        background: "rgba(255,255,255,0.07)",
                        border: "1px solid rgba(255,255,255,0.12)",
                    }}
                />
                {query && (
                    <button
                        onClick={() => setQuery("")}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-white transition-colors"
                        aria-label="Clear search"
                    >
                        ✕
                    </button>
                )}
            </div>

            {/* Genre filter pills */}
            <div className="flex flex-wrap gap-2 mb-8">
                <button
                    onClick={() => setActiveGenre(null)}
                    className="text-sm px-3 py-1.5 rounded-full border transition-colors"
                    style={{
                        borderColor: activeGenre === null ? "var(--netflix-red)" : "rgba(255,255,255,0.15)",
                        color: activeGenre === null ? "var(--netflix-red)" : "#a1a1aa",
                        background: activeGenre === null ? "rgba(229,9,20,0.08)" : "transparent",
                    }}
                >
                    All
                </button>
                {genres.map((g) => (
                    <button
                        key={g}
                        onClick={() => setActiveGenre(activeGenre === g ? null : g)}
                        className="text-sm px-3 py-1.5 rounded-full border transition-colors"
                        style={{
                            borderColor: activeGenre === g ? "var(--netflix-red)" : "rgba(255,255,255,0.15)",
                            color: activeGenre === g ? "var(--netflix-red)" : "#a1a1aa",
                            background: activeGenre === g ? "rgba(229,9,20,0.08)" : "transparent",
                        }}
                    >
                        {g}
                    </button>
                ))}
            </div>

            {/* Grid */}
            {filtered.length === 0 ? (
                <div className="py-20 text-center text-zinc-500">
                    No movies match your search
                    {activeGenre && ` in "${activeGenre}"`}
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

