'use client'

import type { Movie } from '@/lib/api'
import { fetchPopularMovies, upsertReview } from '@/lib/api'
import { getToken, isLoggedIn } from '@/lib/authStore'
import { trackKpi } from '@/lib/kpi'
import { useRouter } from 'next/navigation'
import { useEffect, useMemo, useState } from 'react'

function Stars({ value, onChange }: { value: number; onChange: (v: number) => void }) {
    return (
        <div className='flex gap-1'>
            {[1, 2, 3, 4, 5].map((s) => (
                <button
                    key={s}
                    onClick={() => onChange(s)}
                    className='text-xl leading-none hover:scale-110 transition-transform'
                    style={{ color: s <= value ? '#f59e0b' : 'rgba(255,255,255,0.2)' }}
                >
                    ★
                </button>
            ))}
        </div>
    )
}

export default function OnboardingPage() {
    const router = useRouter()
    const [movies, setMovies] = useState<Movie[]>([])
    const [ratings, setRatings] = useState<Record<number, number>>({})
    const [saving, setSaving] = useState(false)

    useEffect(() => {
        if (!isLoggedIn()) {
            router.replace('/login')
            return
        }
        fetchPopularMovies(15, 0).then((d) => setMovies(d.movies.slice(0, 5)))
    }, [router])

    const ratedCount = useMemo(() => Object.values(ratings).filter((v) => v > 0).length, [ratings])

    const saveAll = async () => {
        const token = getToken()
        if (!token) {
            router.replace('/login')
            return
        }
        setSaving(true)
        for (const m of movies) {
            const r = ratings[m.id]
            if (!r) continue
            await upsertReview(token, m.id, m.title, r, '')
            await trackKpi('rating_submit', 'onboarding', m.id)
        }
        setSaving(false)
        router.push('/')
    }

    return (
        <div className='min-h-screen pt-24 pb-16 px-4 md:px-8' style={{ background: 'var(--bg-primary)' }}>
            <div className='max-w-4xl mx-auto'>
                <h1 className='text-3xl font-black text-white'>Настроим рекомендации за 30 секунд</h1>
                <p className='text-zinc-400 mt-2'>Оцените 5 фильмов, чтобы персонализация быстрее вышла из cold-start.</p>

                <div className='mt-6 grid grid-cols-1 md:grid-cols-2 gap-4'>
                    {movies.map((m) => (
                        <div
                            key={m.id}
                            className='rounded-2xl p-4'
                            style={{ background: 'var(--bg-card)', border: '1px solid rgba(255,255,255,0.08)' }}
                        >
                            <p className='text-white font-semibold'>{m.title}</p>
                            <p className='text-xs text-zinc-500 mt-1'>{m.year ?? '—'} · {(m.genres ?? '').split('|').slice(0, 2).join(', ')}</p>
                            <div className='mt-3'>
                                <Stars
                                    value={ratings[m.id] ?? 0}
                                    onChange={(v) => setRatings((prev) => ({ ...prev, [m.id]: v }))}
                                />
                            </div>
                        </div>
                    ))}
                </div>

                <div className='mt-6 flex items-center gap-3'>
                    <button
                        onClick={saveAll}
                        disabled={saving || ratedCount === 0}
                        className='px-5 py-2.5 rounded-full text-sm font-bold text-white disabled:opacity-40'
                        style={{ background: 'var(--netflix-red)' }}
                    >
                        {saving ? 'Saving...' : `Continue (${ratedCount}/5 rated)`}
                    </button>
                    <button
                        onClick={() => router.push('/')}
                        className='px-5 py-2.5 rounded-full text-sm font-semibold text-zinc-300 border'
                        style={{ borderColor: 'rgba(255,255,255,0.2)' }}
                    >
                        Skip for now
                    </button>
                </div>
            </div>
        </div>
    )
}
