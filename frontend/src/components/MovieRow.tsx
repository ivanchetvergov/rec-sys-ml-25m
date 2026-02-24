"use client";

import { MovieCard } from "@/components/MovieCard";
import type { Movie } from "@/lib/api";
import { useRef } from "react";

interface Props {
    title: string;
    badge?: string;
    movies: Movie[];
    showRank?: boolean;
    onSelect?: (movie: Movie) => void;
}

export function MovieRow({ title, badge, movies, showRank = false, onSelect }: Props) {
    const rowRef = useRef<HTMLDivElement>(null);

    const scroll = (direction: "left" | "right") => {
        if (!rowRef.current) return;
        const amount = rowRef.current.clientWidth * 0.75;
        rowRef.current.scrollBy({ left: direction === "right" ? amount : -amount, behavior: "smooth" });
    };

    return (
        <section className="mb-10">
            {/* Section header */}
            <div className="flex items-center gap-3 mb-3 px-12">
                <h2 className="text-lg font-bold text-white">{title}</h2>
                {badge && (
                    <span className="text-xs font-bold px-2 py-0.5 rounded"
                        style={{ background: "var(--netflix-red)", color: "#fff" }}>
                        {badge}
                    </span>
                )}
            </div>

            {/* Row with scroll buttons */}
            <div className="relative group movie-row">
                {/* Left arrow */}
                <button
                    onClick={() => scroll("left")}
                    className="row-scroll-btn left-0 opacity-0 group-hover:opacity-100"
                    aria-label="Scroll left"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 text-white" fill="none"
                        viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                    </svg>
                </button>

                {/* Cards container */}
                <div
                    ref={rowRef}
                    className="flex gap-2 overflow-x-auto scrollbar-hide px-12"
                    style={{ scrollSnapType: "x mandatory" }}
                >
                    {movies.map((movie, i) => (
                        <div key={movie.id} style={{ scrollSnapAlign: "start", flexShrink: 0 }}>
                            <MovieCard movie={movie} rank={showRank ? i + 1 : undefined} onSelect={onSelect} />
                        </div>
                    ))}
                </div>

                {/* Right arrow */}
                <button
                    onClick={() => scroll("right")}
                    className="row-scroll-btn right-0 opacity-0 group-hover:opacity-100"
                    aria-label="Scroll right"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6 text-white" fill="none"
                        viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                    </svg>
                </button>
            </div>
        </section>
    );
}
