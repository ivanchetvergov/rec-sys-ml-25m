"use client";

import type { Movie, MovieDetails } from "@/lib/api";
import { fetchMovieDetails } from "@/lib/api";
import { useCallback, useEffect, useState } from "react";

interface Props {
    movie: Movie;
    onClose: () => void;
}

const STAR_COUNT = 5;

function StarRating({
    value,
    onChange,
}: {
    value: number;
    onChange: (v: number) => void;
}) {
    const [hovered, setHovered] = useState(0);
    return (
        <div className="flex gap-1">
            {Array.from({ length: STAR_COUNT }, (_, i) => i + 1).map((star) => (
                <button
                    key={star}
                    type="button"
                    onMouseEnter={() => setHovered(star)}
                    onMouseLeave={() => setHovered(0)}
                    onClick={() => onChange(star)}
                    aria-label={`Rate ${star} out of ${STAR_COUNT}`}
                    className="text-2xl transition-transform hover:scale-110 focus:outline-none"
                    style={{
                        color: star <= (hovered || value) ? "#f59e0b" : "rgba(255,255,255,0.2)",
                    }}
                >
                    ★
                </button>
            ))}
        </div>
    );
}

export function MovieDetailModal({ movie, onClose }: Props) {
    const [details, setDetails] = useState<MovieDetails | null>(null);
    const [loading, setLoading] = useState(true);
    const [userRating, setUserRating] = useState(0);
    const [review, setReview] = useState("");
    const [saved, setSaved] = useState(false);

    const storageKey = `movie_review_${movie.id}`;

    // Load TMDB details
    useEffect(() => {
        setLoading(true);
        fetchMovieDetails(movie.id)
            .then(setDetails)
            .finally(() => setLoading(false));
    }, [movie.id]);

    // Load saved review from localStorage
    useEffect(() => {
        try {
            const stored = localStorage.getItem(storageKey);
            if (stored) {
                const parsed = JSON.parse(stored);
                setUserRating(parsed.rating ?? 0);
                setReview(parsed.review ?? "");
            }
        } catch {
            // noop
        }
    }, [storageKey]);

    // Close on Escape
    const handleKeyDown = useCallback(
        (e: KeyboardEvent) => {
            if (e.key === "Escape") onClose();
        },
        [onClose]
    );
    useEffect(() => {
        document.addEventListener("keydown", handleKeyDown);
        return () => document.removeEventListener("keydown", handleKeyDown);
    }, [handleKeyDown]);

    const handleSave = () => {
        try {
            localStorage.setItem(storageKey, JSON.stringify({ rating: userRating, review }));
            setSaved(true);
            setTimeout(() => setSaved(false), 2000);
        } catch {
            // noop
        }
    };

    const genres = movie.genres?.split("|") ?? [];
    const posterUrl = details?.poster_url;
    const gradient = `linear-gradient(135deg, hsl(${(movie.id * 37) % 360},40%,25%), hsl(${(movie.id * 37 + 120) % 360},35%,15%))`;

    return (
        /* Backdrop */
        <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4"
            style={{ background: "rgba(0,0,0,0.85)", backdropFilter: "blur(4px)" }}
            onClick={onClose}
        >
            {/* Panel */}
            <div
                className="relative w-full max-w-3xl max-h-[90vh] overflow-y-auto rounded-xl shadow-2xl"
                style={{ background: "#181818" }}
                onClick={(e) => e.stopPropagation()}
            >
                {/* Close button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 z-10 w-8 h-8 flex items-center justify-center rounded-full text-white"
                    style={{ background: "rgba(0,0,0,0.6)" }}
                    aria-label="Close"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>

                <div className="flex flex-col md:flex-row">
                    {/* Poster */}
                    <div className="md:w-64 flex-shrink-0">
                        {loading ? (
                            <div
                                className="w-full h-72 md:h-full md:min-h-[420px] md:rounded-l-xl animate-pulse"
                                style={{ background: "rgba(255,255,255,0.05)" }}
                            />
                        ) : posterUrl ? (
                            <img
                                src={posterUrl}
                                alt={movie.title}
                                className="w-full h-72 md:h-full object-cover md:rounded-l-xl"
                                style={{ minHeight: 320 }}
                            />
                        ) : (
                            <div
                                className="w-full h-72 md:h-full md:min-h-[420px] flex items-end pb-6 px-4 md:rounded-l-xl"
                                style={{ background: gradient }}
                            >
                                <span className="text-white font-bold text-lg leading-snug drop-shadow-lg">{movie.title}</span>
                            </div>
                        )}
                    </div>

                    {/* Content */}
                    <div className="flex-1 p-6 flex flex-col gap-4">
                        {/* Title & meta */}
                        <div>
                            <h2 className="text-2xl font-bold text-white leading-tight">{movie.title}</h2>
                            {details?.tagline && (
                                <p className="text-sm italic mt-1" style={{ color: "var(--netflix-red)" }}>
                                    "{details.tagline}"
                                </p>
                            )}
                            <div className="flex flex-wrap gap-3 mt-2 text-sm text-zinc-400">
                                {movie.year && <span>{movie.year}</span>}
                                {details?.runtime && <span>{details.runtime} min</span>}
                                {details?.release_date && <span>{details.release_date.slice(0, 4)}</span>}
                            </div>
                        </div>

                        {/* Genres */}
                        {genres.length > 0 && (
                            <div className="flex flex-wrap gap-2">
                                {genres.map((g) => (
                                    <span
                                        key={g}
                                        className="text-xs px-2.5 py-1 rounded-full font-medium"
                                        style={{ background: "rgba(255,255,255,0.08)", color: "#d4d4d8" }}
                                    >
                                        {g}
                                    </span>
                                ))}
                            </div>
                        )}

                        {/* Overview */}
                        {details?.overview && (
                            <p className="text-sm leading-relaxed text-zinc-300">{details.overview}</p>
                        )}

                        {/* Ratings row */}
                        <div className="flex flex-wrap gap-5 text-sm">
                            {movie.avg_rating != null && (
                                <div className="flex flex-col">
                                    <span className="text-zinc-500 text-xs uppercase tracking-wide">Avg rating</span>
                                    <span className="text-white font-bold text-lg">{movie.avg_rating.toFixed(1)}</span>
                                    <span className="text-zinc-500 text-xs">{movie.num_ratings?.toLocaleString()} reviews</span>
                                </div>
                            )}
                            {details?.tmdb_rating != null && (
                                <div className="flex flex-col">
                                    <span className="text-zinc-500 text-xs uppercase tracking-wide">TMDB</span>
                                    <span className="text-white font-bold text-lg">{details.tmdb_rating.toFixed(1)}</span>
                                    <span className="text-zinc-500 text-xs">{details.tmdb_votes?.toLocaleString()} votes</span>
                                </div>
                            )}
                        </div>

                        {/* Divider */}
                        <div style={{ height: 1, background: "rgba(255,255,255,0.08)" }} />

                        {/* User rating */}
                        <div>
                            <p className="text-xs uppercase tracking-wide text-zinc-500 mb-2">Your rating</p>
                            <StarRating value={userRating} onChange={setUserRating} />
                        </div>

                        {/* Review textarea */}
                        <div>
                            <p className="text-xs uppercase tracking-wide text-zinc-500 mb-2">Your review</p>
                            <textarea
                                value={review}
                                onChange={(e) => setReview(e.target.value)}
                                placeholder="Write a few words about the movie..."
                                rows={3}
                                className="w-full text-sm text-zinc-300 rounded-lg px-3 py-2.5 resize-none outline-none focus:ring-1 placeholder-zinc-600"
                                style={{
                                    background: "rgba(255,255,255,0.06)",
                                    border: "1px solid rgba(255,255,255,0.1)",
                                    // @ts-ignore
                                    "--tw-ring-color": "var(--netflix-red)",
                                }}
                            />
                        </div>

                        {/* Save button */}
                        <div className="flex items-center gap-3">
                            <button
                                onClick={handleSave}
                                disabled={userRating === 0 && review.trim() === ""}
                                className="px-5 py-2 rounded font-semibold text-sm text-white transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
                                style={{ background: "var(--netflix-red)" }}
                            >
                                Save
                            </button>
                            {saved && (
                                <span className="text-sm text-green-400 animate-pulse">Saved!</span>
                            )}
                        </div>

                        {/* External links */}
                        <div className="flex gap-4 mt-auto pt-2">
                            {movie.imdb_id && (
                                <a
                                    href={`https://www.imdb.com/title/tt${String(movie.imdb_id).padStart(7, "0")}/`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-xs text-zinc-500 hover:text-yellow-400 transition-colors"
                                >
                                    IMDB ↗
                                </a>
                            )}
                            {movie.tmdb_id && (
                                <a
                                    href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-xs text-zinc-500 hover:text-blue-400 transition-colors"
                                >
                                    TMDB ↗
                                </a>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
