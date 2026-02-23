import type { Movie } from "@/lib/api";

interface Props {
    movie: Movie;
    rank: number;
}

export function MovieCard({ movie, rank }: Props) {
    const genres = movie.genres?.split("|").slice(0, 3) ?? [];

    return (
        <div className="bg-zinc-900 rounded-xl p-4 flex flex-col gap-3 hover:bg-zinc-800 transition-colors">
            {/* Rank + title */}
            <div className="flex items-start gap-3">
                <span className="text-3xl font-black text-zinc-700 leading-none w-8 shrink-0">
                    {rank}
                </span>
                <div>
                    <h3 className="font-semibold leading-snug line-clamp-2">{movie.title}</h3>
                    {movie.year && <p className="text-zinc-500 text-sm mt-0.5">{movie.year}</p>}
                </div>
            </div>

            {/* Genres */}
            {genres.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                    {genres.map((g) => (
                        <span key={g} className="text-xs bg-zinc-800 text-zinc-400 px-2 py-0.5 rounded-full">
                            {g}
                        </span>
                    ))}
                </div>
            )}

            {/* Stats */}
            <div className="flex items-center gap-4 text-sm mt-auto">
                {movie.avg_rating != null && (
                    <span className="text-yellow-400 font-medium">★ {movie.avg_rating.toFixed(1)}</span>
                )}
                {movie.num_ratings != null && (
                    <span className="text-zinc-500">{(movie.num_ratings / 1000).toFixed(0)}k ratings</span>
                )}
            </div>
        </div>
    );
}
