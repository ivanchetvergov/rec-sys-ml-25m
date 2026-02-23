import { MovieCard } from "@/components/MovieCard";
import { fetchPopularMovies } from "@/lib/api";

export default async function HomePage() {
    const data = await fetchPopularMovies(40);

    return (
        <div>
            <h1 className="text-2xl font-bold mb-1">Most Popular</h1>
            <p className="text-zinc-500 text-sm mb-6">
                Ranked by popularity score: avg rating × log(1 + number of ratings)
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {data.movies.map((movie, i) => (
                    <MovieCard key={movie.id} movie={movie} rank={i + 1} />
                ))}
            </div>
        </div>
    );
}
