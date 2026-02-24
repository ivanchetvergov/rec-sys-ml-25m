import { fetchMovie, fetchMovieDetails } from "@/lib/api";
import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

interface PageProps {
    params: { id: string };
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
    const movie = await fetchMovie(Number(params.id));
    return { title: movie ? `${movie.title} — RecSys` : "Movie — RecSys" };
}

export default async function MoviePage({ params }: PageProps) {
    const id = Number(params.id);
    const [movie, details] = await Promise.all([fetchMovie(id), fetchMovieDetails(id)]);

    if (!movie) notFound();

    const genres = movie.genres?.split("|") ?? [];
    const gradient = `linear-gradient(135deg, hsl(${(id * 37) % 360},40%,20%), hsl(${(id * 37 + 120) % 360},35%,12%))`;
    const posterUrl = details?.poster_url;

    return (
        <div className="min-h-screen" style={{ background: "var(--bg-primary)", color: "#fff" }}>
            {/* ── Hero banner ──────────────────────────────────────────────── */}
            <div
                className="relative w-full flex items-end"
                style={{
                    minHeight: "55vh",
                    background: posterUrl
                        ? `linear-gradient(to bottom, rgba(0,0,0,0.3) 0%, var(--bg-primary) 100%), url(${posterUrl}) center/cover no-repeat`
                        : gradient,
                }}
            >
                {/* Bottom fade */}
                <div
                    className="absolute bottom-0 left-0 right-0 h-48 pointer-events-none"
                    style={{ background: "linear-gradient(to bottom, transparent, var(--bg-primary))" }}
                />

                {/* Back button */}
                <div className="absolute top-20 left-0 right-0 mx-auto flex" style={{ maxWidth: 1200, padding: "0 32px" }}>
                    <Link
                        href="/"
                        className="flex items-center gap-2 text-sm text-zinc-300 hover:text-white transition-colors"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
                        </svg>
                        Back
                    </Link>
                </div>
            </div>

            {/* ── Content ──────────────────────────────────────────────────── */}
            <div className="mx-auto px-8 pb-20 mt-[-40px] relative z-10" style={{ maxWidth: 1200 }}>
                <div className="flex flex-col md:flex-row gap-10">
                    {/* Poster column */}
                    <div className="flex-shrink-0 md:w-64">
                        <div
                            className="rounded-2xl overflow-hidden shadow-2xl"
                            style={{
                                aspectRatio: "2/3",
                                background: gradient,
                                border: "1px solid rgba(255,255,255,0.10)",
                            }}
                        >
                            {posterUrl && (
                                <img
                                    src={posterUrl}
                                    alt={movie.title}
                                    className="w-full h-full object-cover"
                                />
                            )}
                        </div>
                    </div>

                    {/* Info column */}
                    <div className="flex-1 pt-2 flex flex-col gap-5">
                        {/* Title */}
                        <div>
                            <h1 className="text-4xl font-black leading-tight mb-1">{movie.title}</h1>
                            {details?.tagline && (
                                <p className="text-base italic" style={{ color: "var(--netflix-red)" }}>
                                    "{details.tagline}"
                                </p>
                            )}
                        </div>

                        {/* Meta row */}
                        <div className="flex flex-wrap items-center gap-4 text-sm text-zinc-400">
                            {movie.year && <span className="font-medium text-zinc-200">{movie.year}</span>}
                            {details?.runtime && <span>{details.runtime} min</span>}
                            {details?.release_date && <span>{details.release_date}</span>}
                        </div>

                        {/* Genres */}
                        {genres.length > 0 && (
                            <div className="flex flex-wrap gap-2">
                                {genres.map((g) => (
                                    <span
                                        key={g}
                                        className="text-sm px-3 py-1 rounded-full font-medium"
                                        style={{ background: "rgba(255,255,255,0.08)", color: "#d4d4d8", border: "1px solid rgba(255,255,255,0.10)" }}
                                    >
                                        {g}
                                    </span>
                                ))}
                            </div>
                        )}

                        {/* Overview */}
                        {details?.overview && (
                            <p className="text-base leading-relaxed text-zinc-300 max-w-2xl">
                                {details.overview}
                            </p>
                        )}

                        {/* Ratings */}
                        <div
                            className="flex flex-wrap gap-8 px-6 py-4 rounded-xl"
                            style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)" }}
                        >
                            {movie.avg_rating != null && (
                                <div>
                                    <p className="text-xs uppercase tracking-wider text-zinc-500 mb-1">Avg rating</p>
                                    <p className="text-3xl font-black text-yellow-400">{movie.avg_rating.toFixed(1)}</p>
                                    <p className="text-xs text-zinc-500 mt-0.5">{movie.num_ratings?.toLocaleString()} reviews</p>
                                </div>
                            )}
                            {details?.tmdb_rating != null && (
                                <div>
                                    <p className="text-xs uppercase tracking-wider text-zinc-500 mb-1">TMDB score</p>
                                    <p className="text-3xl font-black text-blue-400">{details.tmdb_rating.toFixed(1)}</p>
                                    <p className="text-xs text-zinc-500 mt-0.5">{details.tmdb_votes?.toLocaleString()} votes</p>
                                </div>
                            )}
                            {movie.popularity_score != null && (
                                <div>
                                    <p className="text-xs uppercase tracking-wider text-zinc-500 mb-1">Popularity</p>
                                    <p className="text-3xl font-black text-white">{movie.popularity_score.toFixed(2)}</p>
                                    <p className="text-xs text-zinc-500 mt-0.5">internal score</p>
                                </div>
                            )}
                        </div>

                        {/* External links */}
                        <div className="flex gap-4 pt-2">
                            {movie.imdb_id && (
                                <a
                                    href={`https://www.imdb.com/title/tt${String(movie.imdb_id).padStart(7, "0")}/`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="px-5 py-2.5 rounded-lg text-sm font-semibold transition-colors"
                                    style={{ background: "#f5c518", color: "#000" }}
                                >
                                    View on IMDB
                                </a>
                            )}
                            {movie.tmdb_id && (
                                <a
                                    href={`https://www.themoviedb.org/movie/${movie.tmdb_id}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="px-5 py-2.5 rounded-lg text-sm font-semibold text-white transition-opacity hover:opacity-80"
                                    style={{ background: "#01b4e4" }}
                                >
                                    View on TMDB
                                </a>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
