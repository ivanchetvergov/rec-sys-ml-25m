import type { Movie } from "@/lib/api";

interface Props {
    movie: Movie;
    rank?: number;
}

const CARD_GRADIENTS = [
    "135deg, #0f2027 0%, #203a43 50%, #2c5364 100%",
    "135deg, #200122 0%, #6f0000 100%",
    "135deg, #0a3d0c 0%, #0b6e4f 100%",
    "135deg, #2d1b69 0%, #11998e 100%",
    "135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%",
    "135deg, #3a1c71 0%, #d76d77 50%, #ffaf7b 100%",
    "135deg, #134e5e 0%, #71b280 100%",
    "135deg, #373b44 0%, #4286f4 100%",
];

export function MovieCard({ movie, rank }: Props) {
    const genres = movie.genres?.split("|").slice(0, 2) ?? [];
    const gradient = CARD_GRADIENTS[movie.id % CARD_GRADIENTS.length];

    return (
        <div
            className="relative rounded overflow-hidden cursor-pointer group"
            style={{ width: 220 }}
        >
            {/* Poster placeholder — 2:3 aspect ratio */}
            <div
                className="w-full relative"
                style={{
                    aspectRatio: "2/3",
                    background: `linear-gradient(${gradient})`,
                }}
            >
                {/* Rank badge */}
                {rank != null && (
                    <span
                        className="absolute top-2 left-2 text-xs font-bold px-1.5 py-0.5 rounded"
                        style={{ background: "rgba(0,0,0,0.7)", color: "#e5e5e5" }}
                    >
                        #{rank}
                    </span>
                )}

                {/* Rating badge */}
                {movie.avg_rating != null && (
                    <span
                        className="absolute top-2 right-2 text-xs font-bold px-1.5 py-0.5 rounded"
                        style={{ background: "rgba(0,0,0,0.7)", color: "#facc15" }}
                    >
                        ★ {movie.avg_rating.toFixed(1)}
                    </span>
                )}

                {/* Hover overlay */}
                <div
                    className="absolute inset-0 flex flex-col justify-end p-3 opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                    style={{ background: "linear-gradient(to top, rgba(0,0,0,0.95) 0%, rgba(0,0,0,0.3) 60%, transparent 100%)" }}
                >
                    <div className="flex gap-1 mb-1 flex-wrap">
                        {genres.map((g) => (
                            <span key={g} className="text-xs text-zinc-300 bg-white/10 px-1.5 py-0.5 rounded">
                                {g}
                            </span>
                        ))}
                    </div>
                    {movie.num_ratings != null && (
                        <span className="text-xs text-zinc-400">
                            {(movie.num_ratings / 1000).toFixed(0)}k ratings
                        </span>
                    )}
                </div>
            </div>

            {/* Title bar */}
            <div
                className="px-2 py-2"
                style={{ background: "var(--bg-card)" }}
            >
                <p className="text-sm font-semibold text-white line-clamp-1 leading-snug">{movie.title}</p>
                {movie.year && (
                    <p className="text-xs text-zinc-500 mt-0.5">{movie.year}</p>
                )}
            </div>
        </div>
    );
}
