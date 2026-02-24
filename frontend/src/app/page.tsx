import { CatalogSection } from "@/components/CatalogSection";
import { HeroSection } from "@/components/HeroSection";
import { MovieRow } from "@/components/MovieRow";
import { fetchPopularMovies } from "@/lib/api";

export default async function HomePage() {
    // Single fetch — sliced into sections client-side
    const data = await fetchPopularMovies(80);
    const movies = data.movies;

    const hero = movies[0];
    const trending = movies.slice(1, 21);
    // Shuffle positions 20–59 to mimic "personalised" picks
    const recommended = movies.slice(20, 50).sort((a, b) => a.id % 7 - b.id % 7);
    const catalog = movies.slice(40, 80);

    return (
        <div style={{ background: "var(--bg-primary)" }}>
            {/* ── Layer 1: Hero ─────────────────────────────────────────── */}
            {hero && <HeroSection movie={hero} rank={1} />}

            {/* ── Rows section ──────────────────────────────────────────── */}
            <div className="mt-[-80px] relative z-10 pb-4">
                <MovieRow
                    title="Trending Now"
                    badge="TOP 20"
                    movies={trending}
                    showRank
                />
            </div>

            {/* ── Layer 2: Personal Recommendations ─────────────────────── */}
            <section id="popular" className="pb-4">
                <MovieRow
                    title="Recommended for You"
                    badge="NEW"
                    movies={recommended}
                />
            </section>

            {/* ── Layer 3: Catalog ──────────────────────────────────────── */}
            <CatalogSection movies={catalog} />
        </div>
    );
}

