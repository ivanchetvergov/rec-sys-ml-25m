"use client";

import { MovieCard } from "@/components/MovieCard";
import type { Movie } from "@/lib/api";

interface Props {
    movies: Movie[];
    onSelect?: (movie: Movie) => void;
}

export function CatalogSection({ movies, onSelect }: Props) {
    return (
        <section id="catalog" className="px-12 pb-20">
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-xl font-bold text-white">Browse Catalog</h2>
                    <p className="text-sm text-zinc-500 mt-0.5">Explore our full movie library</p>
                </div>
                {/* Filters placeholder */}
                <div className="hidden md:flex items-center gap-2">
                    {["All", "Action", "Drama", "Comedy", "Sci-Fi"].map((label) => (
                        <button
                            key={label}
                            className="text-sm px-3 py-1.5 rounded-full border transition-colors"
                            style={{
                                borderColor: label === "All" ? "var(--netflix-red)" : "rgba(255,255,255,0.15)",
                                color: label === "All" ? "var(--netflix-red)" : "#a1a1aa",
                            }}
                        >
                            {label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Search bar placeholder */}
            <div className="relative mb-8 max-w-md">
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
                    placeholder="Search movies, genres..."
                    disabled
                    className="w-full pl-9 pr-4 py-2.5 rounded-lg text-sm text-zinc-400 outline-none cursor-not-allowed"
                    style={{
                        background: "rgba(255,255,255,0.06)",
                        border: "1px solid rgba(255,255,255,0.1)",
                    }}
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-zinc-600 font-medium">
                    Coming soon
                </span>
            </div>

            {/* Grid */}
            <div
                className="grid gap-3"
                style={{
                    gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                }}
            >
                {movies.map((movie, i) => (
                    <MovieCard key={movie.id} movie={movie} rank={i + 1} onSelect={onSelect} />
                ))}
            </div>

            {/* Load more placeholder */}
            <div className="mt-10 flex justify-center">
                <button
                    disabled
                    className="px-8 py-3 rounded font-semibold text-sm text-zinc-500 cursor-not-allowed"
                    style={{ background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)" }}
                >
                    Load more — coming soon
                </button>
            </div>
        </section>
    );
}
