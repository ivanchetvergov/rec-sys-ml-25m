'use client'

import { useTranslation } from '@/lib/useTranslation'
import Link from 'next/link'

export function BackButton() {
	const { tr } = useTranslation()
	return (
		<Link href='/' className='flex items-center gap-2 text-sm text-zinc-300 hover:text-white transition-colors'>
			<svg xmlns='http://www.w3.org/2000/svg' className='w-4 h-4' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={2.5}>
				<path strokeLinecap='round' strokeLinejoin='round' d='M15 19l-7-7 7-7' />
			</svg>
			{tr.movie.back}
		</Link>
	)
}

export function RatingsBlock({
	avgRating,
	numRatings,
	tmdbRating,
	tmdbVotes,
	popularityScore,
}: {
	avgRating?: number | null
	numRatings?: number | null
	tmdbRating?: number | null
	tmdbVotes?: number | null
	popularityScore?: number | null
}) {
	const { tr } = useTranslation()
	return (
		<div
			className='flex flex-wrap gap-8 px-6 py-4 rounded-xl mt-auto'
			style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' }}
		>
			{avgRating != null && (
				<div>
					<p className='text-xs uppercase tracking-wider text-zinc-500 mb-1'>{tr.movie.avgRating}</p>
					<p className='text-3xl font-black text-yellow-400'>{avgRating.toFixed(1)}</p>
					<p className='text-xs text-zinc-500 mt-0.5'>{numRatings?.toLocaleString()} {tr.movie.ratings}</p>
				</div>
			)}
			{tmdbRating != null && (
				<div>
					<p className='text-xs uppercase tracking-wider text-zinc-500 mb-1'>{tr.movie.tmdbScore}</p>
					<p className='text-3xl font-black text-blue-400'>{tmdbRating.toFixed(1)}</p>
					<p className='text-xs text-zinc-500 mt-0.5'>{tmdbVotes?.toLocaleString()} {tr.movie.votes}</p>
				</div>
			)}
			{popularityScore != null && (
				<div>
					<p className='text-xs uppercase tracking-wider text-zinc-500 mb-1'>{tr.movie.popularity}</p>
					<p className='text-3xl font-black text-white'>{popularityScore.toFixed(2)}</p>
					<p className='text-xs text-zinc-500 mt-0.5'>{tr.movie.internalScore}</p>
				</div>
			)}
		</div>
	)
}
