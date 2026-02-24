import type { Movie } from "@/lib/api";

interface Props {
    movie: Movie;
    rank?: number;
    onSelect?: (movie: Movie) => void;
}

const HERO_GRADIENTS = [
    "135deg, #0f2027, #203a43, #2c5364",
    "135deg, #1a0533, #6a0572, #ab83a1",
    "135deg, #0d1117, #161b22, #1e3a5f",
    "135deg, #200122, #6f0000, #200122",
    "135deg, #0a3d0c, #0b6e4f, #07303e",
    "135deg, #2d1b69, #11998e, #38ef7d",
];

function getGradient(id: number): string {
    return HERO_GRADIENTS[id % HERO_GRADIENTS.length];
}

export function HeroSection({ movie, rank = 1, onSelect }: Props) {
    const genres = movie.genres?.split("|") ?? [];
    const gradient = getGradient(movie.id);

    return (
        <section
            className="relative w-full flex items-end pb-20"
            style={{
                minHeight: "82vh",
                background: `linear-gradient(${gradient})`,
            }}
        >
            {/* Noise overlay for film texture */}
            <div
                className="absolute inset-0"
                style={{
                    background: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E\")",
                    pointerEvents: "none",
                }}
            />

            {/* Bottom fade to dark */}
            <div
                className="absolute bottom-0 left-0 right-0 h-48"
                style={{ background: "linear-gradient(to bottom, transparent, #141414)" }}
            />

            {/* Left fade */}
            <div
                className="absolute inset-y-0 left-0 w-1/3"
                style={{ background: "linear-gradient(to right, rgba(0,0,0,0.4), transparent)" }}
            />

            {/* Content */}
            <div className="relative z-10 px-12 max-w-2xl">
                {/* #1 badge */}
                <div className="flex items-center gap-2 mb-4">
                    <span className="text-xs font-bold tracking-[0.2em] text-zinc-400 uppercase">
                        #{rank} Most Popular
                    </span>
                </div>

                {/* Title */}
                <h1 className="text-5xl md:text-7xl font-black leading-none mb-3 text-white drop-shadow-lg">
                    {movie.title}
                </h1>

                {/* Meta */}
                <div className="flex items-center gap-4 mb-5 text-sm">
                    {movie.year && (
                        <span className="text-zinc-300">{movie.year}</span>
                    )}
                    {movie.avg_rating != null && (
                        <span className="text-yellow-400 font-semibold">★ {movie.avg_rating.toFixed(1)}</span>
                    )}
                    {movie.num_ratings != null && (
                        <span className="text-zinc-400">
                            {(movie.num_ratings / 1000).toFixed(0)}k ratings
                        </span>
                    )}
                </div>

                {/* Genres */}
                {genres.length > 0 && (
                    <div className="flex flex-wrap gap-2 mb-7">
                        {genres.map((g) => (
                            <span
                                key={g}
                                className="text-xs bg-white/10 backdrop-blur-sm text-white px-3 py-1 rounded-full border border-white/20"
                            >
                                {g}
                            </span>
                        ))}
                    </div>
                )}

                {/* CTA buttons */}
                <div className="flex items-center gap-3">
                    <button
                        className="flex items-center gap-2 px-7 py-3 rounded font-bold text-base text-black transition-opacity hover:opacity-80"
                        style={{ background: "#ffffff" }}
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M8 5v14l11-7z" />
                        </svg>
                        Play
                    </button>
                    <button
                        onClick={() => onSelect?.(movie)}
                        className="flex items-center gap-2 px-7 py-3 rounded font-bold text-base text-white transition-colors hover:bg-white/30"
                        style={{ background: "rgba(109,109,110,0.7)" }}
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none"
                            viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 1 1 0 20A10 10 0 0 1 12 2z" />
                        </svg>
                        More Info
                    </button>
                </div>
            </div>
        </section>
    );
}
