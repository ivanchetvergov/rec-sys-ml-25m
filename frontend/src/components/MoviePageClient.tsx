"use client";

import { CatalogSection } from "@/components/CatalogSection";
import { HeroSection } from "@/components/HeroSection";
import { MovieDetailModal } from "@/components/MovieDetailModal";
import { MovieRow } from "@/components/MovieRow";
import type { Movie, PersonalRec, WatchlistItem } from "@/lib/api";
import { fetchPersonalRecs, fetchWatchlist } from "@/lib/api";
import { getAuthUser, getToken, isLoggedIn } from "@/lib/authStore";
import { ensureKpiSessionStarted, trackKpi } from "@/lib/kpi";
import { useEffect, useRef, useState } from "react";

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

    // "me" = logged-in user id (null if not authenticated)
    const [meId, setMeId] = useState<number | null>(null);
    // true = using "me" mode; false = using custom id
    const [useMe, setUseMe] = useState(true);

    const [customInput, setCustomInput] = useState("");
    const [customId, setCustomId] = useState<number | null>(null);

    const [personalMovies, setPersonalMovies] = useState<Movie[]>([]);
    const [personalModel, setPersonalModel] = useState<string>("");
    const [personalLoading, setPersonalLoading] = useState(true);
    const [watchlistItems, setWatchlistItems] = useState<WatchlistItem[]>([]);
    const [fallbackGenre, setFallbackGenre] = useState<string>("");
    const inputRef = useRef<HTMLInputElement>(null);

    const hero = movies[0];
    const trending = movies.slice(1, 21);

    const personalTrustBadges = Object.fromEntries(
        personalMovies.map((m) => {
            const hot = (m.avg_rating ?? 0) >= 4.3 && (m.num_ratings ?? 0) >= 10000;
            const inWatchlistGenre = watchlistItems.some((w) => {
                if (!w.genres || !m.genres) return false;
                const left = new Set(w.genres.split('|'));
                return m.genres.split('|').some((g) => left.has(g));
            });
            const label = inWatchlistGenre
                ? 'Matches your watchlist taste'
                : hot
                    ? 'Community favorite'
                    : 'Recommended by ML';
            return [m.id, label];
        }),
    ) as Record<number, string>;

    const fallbackGenres = Array.from(
        new Set(
            movies
                .flatMap((m) => (m.genres ? m.genres.split('|') : []))
                .filter(Boolean),
        ),
    ).slice(0, 8);

    const fallbackByGenre = movies
        .filter((m) => !fallbackGenre || (m.genres?.split('|').includes(fallbackGenre) ?? false))
        .sort((a, b) => (b.popularity_score ?? 0) - (a.popularity_score ?? 0))
        .slice(0, 12);

    const watchlistReactivationPicks = personalMovies
        .filter((m) =>
            watchlistItems.some((w) => {
                if (!m.genres || !w.genres) return false;
                const mg = new Set(m.genres.split('|'));
                return w.genres.split('|').some((g) => mg.has(g));
            }),
        )
        .slice(0, 3);

    // Sync meId on mount and on auth-change
    useEffect(() => {
        function sync() {
            const authUser = isLoggedIn() ? getAuthUser() : null;
            setMeId(authUser ? authUser.id : null);
        }
        sync();
        window.addEventListener("auth-change", sync);
        return () => window.removeEventListener("auth-change", sync);
    }, []);

    // Active user id for recommendations
    const userId = useMe ? meId : customId;

    useEffect(() => {
        ensureKpiSessionStarted();
    }, []);

    useEffect(() => {
        if (!fallbackGenre && fallbackGenres.length > 0) {
            setFallbackGenre(fallbackGenres[0]);
        }
    }, [fallbackGenre, fallbackGenres]);

    // Fetch personal recs whenever effective userId changes
    useEffect(() => {
        if (userId === null) {
            setPersonalMovies([]);
            setPersonalModel("");
            setPersonalLoading(false);
            return;
        }
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

    useEffect(() => {
        const token = getToken();
        if (!token) {
            setWatchlistItems([]);
            return;
        }
        fetchWatchlist(token).then(setWatchlistItems);
    }, [meId]);

    const applyCustom = () => {
        const v = parseInt(customInput, 10);
        if (!isNaN(v) && v > 0) {
            setCustomId(v);
            setUseMe(false);
        }
    };

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

                        {/* Me button */}
                        <button
                            onClick={() => setUseMe(true)}
                            disabled={meId === null}
                            className="text-xs px-4 py-1.5 rounded-full border font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                            style={{
                                borderColor: useMe ? "var(--netflix-red)" : "rgba(255,255,255,0.15)",
                                color: useMe ? "var(--netflix-red)" : "#a1a1aa",
                                background: useMe ? "rgba(229,9,20,0.10)" : "transparent",
                            }}
                            title={meId === null ? "Sign in to use your own recommendations" : `User #${meId}`}
                        >
                            Me{meId !== null && useMe && (
                                <span className="ml-1 opacity-50 font-normal">#{meId}</span>
                            )}
                        </button>

                        {/* Custom ID input */}
                        <div
                            className="flex items-center rounded-full border overflow-hidden transition-colors"
                            style={{
                                borderColor: !useMe && customId !== null ? "var(--netflix-red)" : "rgba(255,255,255,0.15)",
                                background: !useMe && customId !== null ? "rgba(229,9,20,0.10)" : "rgba(255,255,255,0.04)",
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
                                className="text-xs bg-transparent outline-none px-1.5 py-1 w-16"
                                style={{ color: !useMe && customId !== null ? "var(--netflix-red)" : "#a1a1aa" }}
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
                                {personalModel.startsWith("two_stage")
                                    ? "iALS + CatBoost Ranker"
                                    : "Popularity fallback"}
                            </span>
                        )}
                    </div>

                    {userId === null ? (
                        <div className="h-48 flex items-center justify-center text-zinc-500 text-sm">
                            Sign in or enter a user ID to see recommendations
                        </div>
                    ) : personalLoading ? (
                        <div className="h-48 flex items-center justify-center text-zinc-500 text-sm">
                            Loading recommendations…
                        </div>
                    ) : personalMovies.length === 0 ? (
                        <div className="flex flex-col gap-6">
                            <div
                                className="rounded-2xl p-5"
                                style={{
                                    background: 'rgba(255,255,255,0.04)',
                                    border: '1px solid rgba(255,255,255,0.08)',
                                }}
                            >
                                <h3 className="text-white text-lg font-bold">No personal picks yet</h3>
                                <p className="text-zinc-400 text-sm mt-1">
                                    Оцените 3 фильма для персонализации, и рекомендации станут заметно точнее.
                                </p>
                                <div className="mt-4 flex flex-wrap gap-2">
                                    {trending.slice(0, 3).map((m) => (
                                        <button
                                            key={m.id}
                                            onClick={() => setSelected(m)}
                                            className="px-3 py-1.5 text-xs rounded-full border text-zinc-200"
                                            style={{ borderColor: 'rgba(255,255,255,0.2)', background: 'rgba(255,255,255,0.08)' }}
                                        >
                                            Rate {m.title}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div>
                                <div className="flex items-center justify-between mb-3">
                                    <h4 className="text-sm font-bold text-white">Popular by genre</h4>
                                    <div className="flex flex-wrap gap-2">
                                        {fallbackGenres.map((g) => (
                                            <button
                                                key={g}
                                                onClick={() => setFallbackGenre(g)}
                                                className="text-xs px-2.5 py-1 rounded-full border"
                                                style={{
                                                    borderColor: fallbackGenre === g ? 'rgba(229,9,20,0.5)' : 'rgba(255,255,255,0.2)',
                                                    color: fallbackGenre === g ? '#f87171' : '#a1a1aa',
                                                    background: fallbackGenre === g ? 'rgba(229,9,20,0.12)' : 'transparent',
                                                }}
                                            >
                                                {g}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <MovieRow
                                    title={`Popular in ${fallbackGenre || 'All genres'}`}
                                    badge="START HERE"
                                    movies={fallbackByGenre}
                                    onSelect={setSelected}
                                />
                            </div>
                        </div>
                    ) : (
                        <>
                            {watchlistItems.length > 0 && (
                                <div
                                    className="rounded-2xl p-4 mb-4"
                                    style={{ background: 'rgba(16,185,129,0.09)', border: '1px solid rgba(16,185,129,0.25)' }}
                                >
                                    <p className="text-sm text-emerald-300 font-semibold">
                                        У вас {watchlistItems.length} фильмов в watchlist. Вот 3 наиболее релевантных сегодня:
                                    </p>
                                    <div className="mt-3 flex flex-wrap gap-2">
                                        {(watchlistReactivationPicks.length > 0 ? watchlistReactivationPicks : personalMovies.slice(0, 3)).map((m) => (
                                            <button
                                                key={m.id}
                                                onClick={() => setSelected(m)}
                                                className="text-xs px-2.5 py-1.5 rounded-full border text-zinc-200"
                                                style={{ borderColor: 'rgba(255,255,255,0.2)', background: 'rgba(0,0,0,0.2)' }}
                                            >
                                                {m.title}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <MovieRow
                                title="Recommended for You"
                                badge={personalModel.startsWith("two_stage") ? "ML" : personalModel === "popularity_fallback" ? "TOP" : undefined}
                                movies={personalMovies}
                                trustBadgeByMovieId={personalTrustBadges}
                                onRowImpression={() => trackKpi('rec_impression', 'personal')}
                                onMovieClick={(m) => trackKpi('rec_click', 'personal', m.id)}
                                onSelect={setSelected}
                            />
                        </>
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
