"use client";

import { CatalogSection } from "@/components/CatalogSection";
import { HeroSection } from "@/components/HeroSection";
import { MovieDetailModal } from "@/components/MovieDetailModal";
import { MovieRow } from "@/components/MovieRow";
import type { Movie, PersonalRec } from "@/lib/api";
import { fetchPersonalRecs } from "@/lib/api";
import { useEffect, useState } from "react";

// Demo user selector — in production this would come from auth context
const DEMO_USERS = [1, 42, 123, 500, 1000, 5000, 10000];

/** Map a PersonalRec to the Movie shape expected by MovieRow / MovieCard. */
function toMovie(r: PersonalRec): Movie {
    return {
        id: r.id,
        title: r.title ?? `Movie ${r.id}`,
        genres: r.genres,
        year: r.year,
        avg_rating: r.avg_rating,
        num_ratings: r.num_ratings,
        popularity_score: r.popularity_score,
        tmdb_id: r.tmdb_id,
        imdb_id: null,
    };
}

interface Props {
    movies: Movie[];
}

export function MoviePageClient({ movies }: Props) {
    const [selected, setSelected] = useState<Movie | null>(null);
    const [userId, setUserId] = useState<number>(DEMO_USERS[0]);
    const [personalMovies, setPersonalMovies] = useState<Movie[]>([]);
    const [personalModel, setPersonalModel] = useState<string>("");
    const [personalLoading, setPersonalLoading] = useState(true);

    const hero = movies[0];
    const trending = movies.slice(1, 21);

    // Fetch personal recs whenever userId changes
    useEffect(() => {
        setPersonalLoading(true);
        fetchPersonalRecs(userId, 24)
            .then((data) => {
                if (data && data.movies.length > 0) {
                    setPersonalMovies(data.movies.map(toMovie));
                    setPersonalModel(data.model);
                } else {
                    setPersonalMovies([]);
                    setPersonalModel("");
                }
            })
            .finally(() => setPersonalLoading(false));
    }, [userId]);

    const personalBadge =
        personalModel === "two_stage" ? "ML" : personalModel === "popularity_fallback" ? "TOP" : undefined;

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

                {/* Personalised row */}
                <section id="personal" className="pb-4">
                    {/* User switcher */}
                    <div className="flex items-center gap-3 mb-3 flex-wrap">
                        <span className="text-sm text-zinc-400">Demo user:</span>
                        {DEMO_USERS.map((uid) => (
                            <button
                                key={uid}
                                onClick={() => setUserId(uid)}
                                className="text-xs px-3 py-1 rounded-full border transition-colors"
                                style={{
                                    borderColor: uid === userId ? "var(--netflix-red)" : "rgba(255,255,255,0.15)",
                                    color: uid === userId ? "var(--netflix-red)" : "#a1a1aa",
                                    background: uid === userId ? "rgba(229,9,20,0.10)" : "transparent",
                                }}
                            >
                                #{uid}
                            </button>
                        ))}
                        {personalModel && (
                            <span className="ml-auto text-xs text-zinc-500">
                                {personalModel === "two_stage"
                                    ? "iALS + CatBoost Ranker"
                                    : "Popularity fallback"}
                            </span>
                        )}
                    </div>

                    {personalLoading ? (
                        <div className="h-48 flex items-center justify-center text-zinc-500 text-sm">
                            Loading recommendations…
                        </div>
                    ) : (
                        <MovieRow
                            title="Recommended for You"
                            badge={personalBadge}
                            movies={personalMovies}
                            onSelect={setSelected}
                        />
                    )}
                </section>

                {/* Catalog — self-fetches the full catalog client-side */}
                <CatalogSection onSelect={setSelected} />
            </div>

            {/* Detail modal */}
            {selected && (
                <MovieDetailModal movie={selected} onClose={() => setSelected(null)} />
            )}
        </div>
    );
}
