"use client";

import { CatalogSection } from "@/components/CatalogSection";
import { HeroSection } from "@/components/HeroSection";
import { MovieDetailModal } from "@/components/MovieDetailModal";
import { MovieRow } from "@/components/MovieRow";
import type { Movie, PersonalRec } from "@/lib/api";
import { fetchPersonalRecs } from "@/lib/api";
import { useEffect, useRef, useState } from "react";

const DEMO_USERS = [1, 42, 123];

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
    const [customInput, setCustomInput] = useState("");
    const [personalMovies, setPersonalMovies] = useState<Movie[]>([]);
    const [personalModel, setPersonalModel] = useState<string>("");
    const [personalLoading, setPersonalLoading] = useState(true);
    const inputRef = useRef<HTMLInputElement>(null);

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

    const isCustomActive = !DEMO_USERS.includes(userId);

    const applyCustom = () => {
        const v = parseInt(customInput, 10);
        if (!isNaN(v) && v > 0) setUserId(v);
    };

    const personalBadge =
        personalModel === "two_stage" ? "ML" : personalModel === "popularity_fallback" ? "TOP" : undefined;

    return (
        <div style={{ background: "var(--bg-primary)" }}>
            {/* Hero */}
            {hero && <HeroSection movie={hero} rank={1} onSelect={setSelected} />}

            {/* Constrained content area */}
            <div className="mx-auto" style={{ maxWidth: 1440, padding: "0 40px" }}>
                {/* Trending row */}
                <div className="mt-[-60px] relative z-10 pb-6">
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
                    <div className="flex items-center gap-2 mb-3 flex-wrap">
                        <span className="text-sm text-zinc-400">Demo user:</span>

                        {/* Preset buttons */}
                        {DEMO_USERS.map((uid) => (
                            <button
                                key={uid}
                                onClick={() => { setUserId(uid); setCustomInput(""); }}
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

                        {/* Custom user input */}
                        <div
                            className="flex items-center rounded-full border overflow-hidden transition-colors"
                            style={{
                                borderColor: isCustomActive ? "var(--netflix-red)" : "rgba(255,255,255,0.15)",
                                background: isCustomActive ? "rgba(229,9,20,0.10)" : "rgba(255,255,255,0.04)",
                            }}
                        >
                            <span className="text-xs pl-3 text-zinc-500">#</span>
                            <input
                                ref={inputRef}
                                type="text"
                                inputMode="numeric"
                                pattern="[0-9]*"
                                value={customInput}
                                onChange={(e) => setCustomInput(e.target.value.replace(/\D/g, ""))}
                                onKeyDown={(e) => e.key === "Enter" && applyCustom()}
                                placeholder="user id"
                                className="text-xs bg-transparent outline-none px-1.5 py-1 w-12"
                                style={{ color: isCustomActive ? "var(--netflix-red)" : "#a1a1aa" }}
                            />
                            <button
                                onClick={applyCustom}
                                className="text-xs px-2.5 py-1 transition-colors hover:text-white"
                                style={{ color: "#a1a1aa" }}
                                aria-label="Apply user id"
                            >
                                →
                            </button>
                        </div>

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
