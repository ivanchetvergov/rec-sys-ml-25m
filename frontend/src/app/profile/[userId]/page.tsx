import {
    fetchPublicUserProfile,
    type Movie,
    type PublicProfileReviewItem,
    type PublicProfileWatchedItem,
    type PublicProfileWatchlistItem,
} from '@/lib/api'
import { AvatarIcon } from '@/lib/avatars'
import Link from 'next/link'
import { notFound } from 'next/navigation'

interface PageProps {
    params: { userId: string }
}

function toMovieFromWatched(item: PublicProfileWatchedItem): Movie {
    return {
        id: item.movie_id,
        title: item.title,
        genres: item.genres,
        year: item.year,
        avg_rating: item.avg_rating,
        num_ratings: item.num_ratings,
        popularity_score: item.popularity_score,
        tmdb_id: item.tmdb_id,
        imdb_id: item.imdb_id,
    }
}

function toMovieFromWatchlist(item: PublicProfileWatchlistItem): Movie {
    return {
        id: item.movie_id,
        title: item.title,
        genres: item.genres,
        year: item.year,
        avg_rating: item.avg_rating,
        num_ratings: item.num_ratings,
        popularity_score: item.popularity_score,
        tmdb_id: item.tmdb_id,
        imdb_id: item.imdb_id,
    }
}

function MovieChip({ movie }: { movie: Movie }) {
    return (
        <Link
            href={`/movies/${movie.id}`}
            className='rounded-lg px-3 py-2 transition-colors hover:bg-white/10'
            style={{ background: 'rgba(255,255,255,0.05)' }}
        >
            <p className='text-sm text-white font-medium'>{movie.title}</p>
            <p className='text-xs text-zinc-500'>
                {[movie.year, movie.genres?.split('|').slice(0, 2).join(', ')].filter(Boolean).join(' · ')}
            </p>
        </Link>
    )
}

function ReviewCard({ item }: { item: PublicProfileReviewItem }) {
    return (
        <div
            className='rounded-xl px-4 py-3'
            style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
        >
            <div className='flex items-center justify-between'>
                <Link href={`/movies/${item.movie_id}`} className='text-sm text-white font-semibold hover:underline'>
                    {item.title}
                </Link>
                <span className='text-xs text-zinc-500'>
                    {new Date(item.created_at).toLocaleDateString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                    })}
                </span>
            </div>
            <p className='text-xs text-yellow-400 mt-1'>{'★'.repeat(item.rating)}</p>
            {item.review_text?.trim() && <p className='text-sm text-zinc-300 mt-2'>{item.review_text}</p>}
        </div>
    )
}

export default async function PublicProfilePage({ params }: PageProps) {
    const userId = Number(params.userId)
    if (!Number.isFinite(userId) || userId <= 0) notFound()

    const profile = await fetchPublicUserProfile(userId, 30)
    if (!profile) notFound()

    return (
        <div className='min-h-screen pt-24 pb-16 px-4 md:px-8' style={{ background: 'var(--bg-primary)' }}>
            <div className='max-w-5xl mx-auto flex flex-col gap-6'>
                <div className='flex items-center gap-4'>
                    <div className='w-14 h-14 rounded-full overflow-hidden' style={{ background: 'rgba(255,255,255,0.06)' }}>
                        <AvatarIcon avatarId={profile.user_avatar_id} size={56} />
                    </div>
                    <div>
                        <h1 className='text-2xl font-black text-white'>{profile.user_login}</h1>
                        <p className='text-sm text-zinc-500'>Public profile</p>
                    </div>
                </div>

                <div className='grid grid-cols-2 sm:grid-cols-4 gap-4'>
                    {[
                        { label: 'Watched', value: profile.watched_count },
                        { label: 'Watchlist', value: profile.watchlist_count },
                        { label: 'Reviews', value: profile.reviews_count },
                        { label: 'Avg rating', value: profile.avg_rating != null ? profile.avg_rating.toFixed(1) : '—' },
                    ].map(s => (
                        <div key={s.label} className='rounded-xl p-4 text-center' style={{ background: 'var(--bg-card)', border: '1px solid rgba(255,255,255,0.08)' }}>
                            <p className='text-2xl font-black text-white'>{s.value}</p>
                            <p className='text-xs text-zinc-500 mt-1'>{s.label}</p>
                        </div>
                    ))}
                </div>

                <div className='grid grid-cols-1 md:grid-cols-2 gap-6'>
                    <div className='rounded-2xl p-5' style={{ background: 'var(--bg-card)', border: '1px solid rgba(255,255,255,0.08)' }}>
                        <h2 className='text-lg font-bold text-white mb-3'>Watched</h2>
                        <div className='flex flex-col gap-2'>
                            {profile.watched.length === 0 ? (
                                <p className='text-sm text-zinc-500'>No watched movies yet.</p>
                            ) : (
                                profile.watched.slice(0, 10).map(item => <MovieChip key={`w-${item.movie_id}`} movie={toMovieFromWatched(item)} />)
                            )}
                        </div>
                    </div>

                    <div className='rounded-2xl p-5' style={{ background: 'var(--bg-card)', border: '1px solid rgba(255,255,255,0.08)' }}>
                        <h2 className='text-lg font-bold text-white mb-3'>Watchlist</h2>
                        <div className='flex flex-col gap-2'>
                            {profile.watchlist.length === 0 ? (
                                <p className='text-sm text-zinc-500'>Watchlist is empty.</p>
                            ) : (
                                profile.watchlist.slice(0, 10).map(item => <MovieChip key={`wl-${item.movie_id}`} movie={toMovieFromWatchlist(item)} />)
                            )}
                        </div>
                    </div>
                </div>

                <div className='rounded-2xl p-5' style={{ background: 'var(--bg-card)', border: '1px solid rgba(255,255,255,0.08)' }}>
                    <h2 className='text-lg font-bold text-white mb-3'>Reviews</h2>
                    <div className='flex flex-col gap-3'>
                        {profile.reviews.length === 0 ? (
                            <p className='text-sm text-zinc-500'>No reviews yet.</p>
                        ) : (
                            profile.reviews.map(item => <ReviewCard key={`r-${item.movie_id}-${item.created_at}`} item={item} />)
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
