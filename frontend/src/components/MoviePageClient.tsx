"use client";

import { CatalogSection } from "@/components/CatalogSection";
import { HeroSection } from "@/components/HeroSection";
import { MovieDetailModal } from "@/components/MovieDetailModal";
import { MovieRow } from "@/components/MovieRow";
import type { Movie } from "@/lib/api";
import { useState } from "react";

interface Props {
    movies: Movie[];
}

export function MoviePageClient({ movies }: Props) {
    const [selected, setSelected] = useState<Movie | null>(null);

    const hero = movies[0];
    const trending = movies.slice(1, 21);
    const recommended = movies.slice(21, 50).sort((a, b) => (a.id % 7) - (b.id % 7));
    const catalog = movies.slice(50);

    return (
        <div style={{ background: "var(--bg-primary)" }}>
            {/* Hero */}
            {hero && <HeroSection movie={hero} rank={1} onSelect={setSelected} />}

            {/* Constrained content area */}
            <div className="mx-auto" style={{ maxWidth: 1440, padding: "0 40px" }}>
                {/* Trending row */}
                <div className="mt-[-80px] relative z-10 pb-4">
                    <MovieRow
                        title="Trending Now"
                        badge="TOP 20"
                        movies={trending}
                        showRank
                        onSelect={setSelected}
                    />
                </div>

                {/* Recommended row */}
                <section id="popular" className="pb-4">
                    <MovieRow
                        title="Recommended for You"
                        badge="NEW"
                        movies={recommended}
                        onSelect={setSelected}
                    />
                </section>

                {/* Catalog */}
                <CatalogSection movies={catalog} onSelect={setSelected} />
            </div>

            {/* Detail modal */}
            {selected && (
                <MovieDetailModal movie={selected} onClose={() => setSelected(null)} />
            )}
        </div>
    );
}
