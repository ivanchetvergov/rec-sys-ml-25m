import { MoviePageClient } from "@/components/MoviePageClient";
import { fetchPopularMovies } from "@/lib/api";

export default async function HomePage() {
    const data = await fetchPopularMovies(21);  // hero (1) + trending (20)
    return <MoviePageClient movies={data.movies} />;
}

