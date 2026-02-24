import { MoviePageClient } from "@/components/MoviePageClient";
import { fetchPopularMovies } from "@/lib/api";

export default async function HomePage() {
    const data = await fetchPopularMovies(100);
    return <MoviePageClient movies={data.movies} />;
}

