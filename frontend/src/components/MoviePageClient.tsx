"use client";

import { CatalogSection } from "@/components/CatalogSection";
import { HeroSection } from "@/components/HeroSection";
import { MovieDetailModal } from "@/components/MovieDetailModal";
import { MovieRow } from "@/components/MovieRow";
import type { Movie, PersonalRec, WatchedItem } from "@/lib/api";
import { fetchPersonalRecs, fetchWatched } from "@/lib/api";
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

type RecommendationSource = 'personal' | 'genre_fallback';

interface RecommendationInsight {
    headline: string;
    confidence: number;
    reasons: string[];
    stats: Array<{ label: string; value: string }>;
    anchors: Array<{
        title: string;
        commonGenres: string[];
        similarity: number;
        year: number | null;
        yearDelta: number | null;
    }>;
}

function splitGenres(genres: string | null | undefined): string[] {
    if (!genres) return [];
    return genres.split('|').map((g) => g.trim()).filter(Boolean);
}

export function MoviePageClient({ movies }: Props) {
    const [selected, setSelected] = useState<Movie | null>(null);
    const [insightMovie, setInsightMovie] = useState<Movie | null>(null);
    const [insightSource, setInsightSource] = useState<RecommendationSource | null>(null);

    // "me" = logged-in user id (null if not authenticated)
    const [meId, setMeId] = useState<number | null>(null);
    // true = using "me" mode; false = using custom id
    const [useMe, setUseMe] = useState(true);

    const [customInput, setCustomInput] = useState("");
    const [customId, setCustomId] = useState<number | null>(null);

    const [personalMovies, setPersonalMovies] = useState<Movie[]>([]);
    const [personalModel, setPersonalModel] = useState<string>("");
    const [personalLoading, setPersonalLoading] = useState(true);
    const [watchedItems, setWatchedItems] = useState<WatchedItem[]>([]);
    const [fallbackGenre, setFallbackGenre] = useState<string>("");
    const inputRef = useRef<HTMLInputElement>(null);

    const hero = movies[0];
    const trending = movies.slice(1, 21);

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

    const openInsight = async (movie: Movie, source: RecommendationSource) => {
        setInsightMovie(movie);
        setInsightSource(source);
        await trackKpi('explanation_open', source, movie.id);
    };

    const buildRecommendationInsight = (movie: Movie, source: RecommendationSource): RecommendationInsight => {
        const movieGenres = splitGenres(movie.genres);
        const watchedGenreCounts = watchedItems.reduce<Record<string, number>>((acc, item) => {
            for (const g of splitGenres(item.genres)) {
                acc[g] = (acc[g] ?? 0) + 1;
            }
            return acc;
        }, {});

        const anchors = watchedItems
            .map((w) => {
                const watchedGenres = splitGenres(w.genres);
                const commonGenres = watchedGenres.filter((g) => movieGenres.includes(g));
                const union = new Set([...watchedGenres, ...movieGenres]);
                const genreJaccard = union.size > 0 ? commonGenres.length / union.size : 0;
                const yearDelta = w.year != null && movie.year != null ? Math.abs(w.year - movie.year) : null;
                const yearSimilarity = yearDelta == null ? 0.5 : Math.max(0, 1 - yearDelta / 30);
                const similarity = Math.round((genreJaccard * 0.8 + yearSimilarity * 0.2) * 100);
                return {
                    title: w.title,
                    commonGenres,
                    similarity,
                    year: w.year,
                    yearDelta,
                };
            })
            .filter((a) => a.commonGenres.length > 0)
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 3);

        const anchorTitles = anchors.map((a) => a.title);
        const matchedGenres = movieGenres.filter((g) => watchedGenreCounts[g] != null);
        const genreHits = matchedGenres.reduce((sum, g) => sum + (watchedGenreCounts[g] ?? 0), 0);
        const topSimilarity = anchors[0]?.similarity ?? 0;
        const meanSimilarity = anchors.length > 0
            ? Math.round(anchors.reduce((sum, a) => sum + a.similarity, 0) / anchors.length)
            : 0;

        const score = movie.popularity_score ?? 0;
        const distribution = movies.map((m) => m.popularity_score ?? 0).sort((a, b) => a - b);
        const lessOrEqual = distribution.filter((v) => v <= score).length;
        const popularityPercentile = distribution.length > 0 ? Math.round((lessOrEqual / distribution.length) * 100) : 0;

        const genreSignal = matchedGenres.length > 0 ? Math.min(1, matchedGenres.length / 3) : 0.2;
        const historySignal = Math.min(1, meanSimilarity / 100);
        const popularitySignal = Math.min(1, popularityPercentile / 100);
        const confidence = Math.round(
            (historySignal * 0.6 + genreSignal * 0.25 + popularitySignal * 0.15) * 100,
        );

        const reasons: string[] = [];
        if (anchorTitles.length > 0) {
            reasons.push(`Вы уже смотрели похожие фильмы: ${anchorTitles.join(', ')}.`);
            reasons.push(`Основа похожести: общие жанры (${matchedGenres.slice(0, 4).join(', ')}), плюс близость по эпохе выпуска.`);
            reasons.push(`В вашей истории найдено ${genreHits} пересечений по жанрам.`);
        } else if (matchedGenres.length > 0) {
            reasons.push(`Есть совпадение с вашими прошлыми просмотрами по жанрам: ${matchedGenres.slice(0, 3).join(', ')}.`);
        } else {
            reasons.push('Фильм выбран как расширение вкуса: добавляет новые жанры при сохранении качества подборки.');
        }

        if (source === 'personal') {
            reasons.push('Рекомендация опирается на ваши реальные просмотры, а не на случайный популярный список.');
        } else {
            reasons.push('Это стартовая рекомендация по жанровой релевантности до накопления персональной истории.');
        }

        return {
            headline: source === 'personal' ? 'Почему это персональная рекомендация' : 'Почему это релевантный стартовый вариант',
            confidence,
            reasons,
            stats: [
                { label: 'Источник сигнала', value: source === 'personal' ? 'Персональная модель' : 'Жанровый fallback' },
                { label: 'Сходство с историей', value: anchors.length > 0 ? `${meanSimilarity}% (топ: ${topSimilarity}%)` : 'Недостаточно данных' },
                { label: 'Популярность (перцентиль)', value: ` ${popularityPercentile}%` },
                { label: 'Опорные просмотренные фильмы', value: anchorTitles.length > 0 ? anchorTitles.join(', ') : 'Не найдены' },
                { label: 'Жанровых матчей', value: `${matchedGenres.length}` },
                { label: 'Модель', value: source === 'personal' && personalModel ? personalModel : 'genre_fallback' },
            ],
            anchors,
        };
    };

    const insight = insightMovie && insightSource ? buildRecommendationInsight(insightMovie, insightSource) : null;

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
            setWatchedItems([]);
            return;
        }
        fetchWatched(token).then(setWatchedItems);
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
                                    movies={fallbackByGenre}
                                    onMovieExplain={(m) => openInsight(m, 'genre_fallback')}
                                    onSelect={setSelected}
                                />
                            </div>
                        </div>
                    ) : (
                        <>
                            <MovieRow
                                title="Recommended for You"
                                movies={personalMovies}
                                onRowImpression={() => trackKpi('rec_impression', 'personal')}
                                onMovieClick={(m) => trackKpi('rec_click', 'personal', m.id)}
                                onMovieExplain={(m) => openInsight(m, 'personal')}
                                onSelect={setSelected}
                            />
                        </>
                    )}

                    {insightMovie && insight && (
                        <div
                            className="rounded-2xl p-5 mt-3"
                            style={{ background: 'rgba(34,211,238,0.08)', border: '1px solid rgba(34,211,238,0.25)' }}
                        >
                            <div className="flex items-start justify-between gap-4">
                                <div>
                                    <h3 className="text-white text-lg font-bold">{insightMovie.title}</h3>
                                    <p className="text-cyan-100/90 text-sm mt-1">{insight.headline}</p>
                                </div>
                                <button
                                    onClick={() => {
                                        setInsightMovie(null);
                                        setInsightSource(null);
                                    }}
                                    className="text-xs px-2.5 py-1 rounded-full border text-cyan-100"
                                    style={{ borderColor: 'rgba(255,255,255,0.3)', background: 'rgba(0,0,0,0.2)' }}
                                >
                                    Закрыть
                                </button>
                            </div>

                            <div className="mt-4">
                                <div className="flex items-center justify-between text-xs text-cyan-50/90 mb-1.5">
                                    <span>Уверенность рекомендации</span>
                                    <span>{insight.confidence}%</span>
                                </div>
                                <div className="h-2 rounded-full" style={{ background: 'rgba(255,255,255,0.12)' }}>
                                    <div
                                        className="h-full rounded-full"
                                        style={{ width: `${insight.confidence}%`, background: 'linear-gradient(90deg, #22d3ee 0%, #14b8a6 100%)' }}
                                    />
                                </div>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-3 gap-2.5 mt-4">
                                {insight.stats.map((s) => (
                                    <div
                                        key={s.label}
                                        className="rounded-xl p-2.5"
                                        style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)' }}
                                    >
                                        <p className="text-[11px] text-cyan-50/75">{s.label}</p>
                                        <p className="text-sm text-white font-semibold mt-0.5">{s.value}</p>
                                    </div>
                                ))}
                            </div>

                            <div className="mt-4 space-y-2.5">
                                {insight.reasons.map((reason) => (
                                    <div
                                        key={reason}
                                        className="text-sm text-zinc-100 rounded-lg px-3 py-2"
                                        style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.08)' }}
                                    >
                                        {reason}
                                    </div>
                                ))}
                            </div>

                            {insight.anchors.length > 0 && (
                                <div className="mt-4">
                                    <p className="text-sm font-semibold text-white mb-2">Похоже на просмотренные вами</p>
                                    <div className="space-y-2">
                                        {insight.anchors.map((a) => (
                                            <div
                                                key={a.title}
                                                className="rounded-lg px-3 py-2.5"
                                                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.12)' }}
                                            >
                                                <div className="flex items-center justify-between gap-3">
                                                    <p className="text-sm text-white font-semibold">{a.title}</p>
                                                    <span className="text-xs text-cyan-200">Сходство: {a.similarity}%</span>
                                                </div>
                                                <p className="text-xs text-zinc-300 mt-1">
                                                    Общие жанры: {a.commonGenres.join(', ')}
                                                    {a.yearDelta != null ? ` • Разница по году: ${a.yearDelta}` : ''}
                                                </p>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
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
