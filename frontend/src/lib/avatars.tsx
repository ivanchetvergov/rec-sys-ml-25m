import type { CSSProperties } from 'react';

export type AvatarId = 'cat' | 'fox' | 'owl' | 'panda' | 'koala'

export const AVATAR_OPTIONS: Array<{ id: AvatarId; label: string }> = [
    { id: 'cat', label: 'Cat' },
    { id: 'fox', label: 'Fox' },
    { id: 'owl', label: 'Owl' },
    { id: 'panda', label: 'Panda' },
    { id: 'koala', label: 'Koala' },
]

export const DEFAULT_AVATAR_ID: AvatarId = 'cat'

export function defaultAvatarForUser(userId: number): AvatarId {
    const idx = Math.abs(userId) % AVATAR_OPTIONS.length
    return AVATAR_OPTIONS[idx].id
}

function iconCommonStyle(size: number): CSSProperties {
    return {
        width: size,
        height: size,
        display: 'block',
    }
}

export function AvatarIcon({
    avatarId,
    size = 36,
}: {
    avatarId: AvatarId
    size?: number
}) {
    switch (avatarId) {
        case 'cat':
            return (
                <svg viewBox='0 0 64 64' style={iconCommonStyle(size)} aria-hidden='true'>
                    <rect width='64' height='64' rx='32' fill='#2b2d42' />
                    <path d='M17 21l8 5 2-11zM47 21l-8 5-2-11z' fill='#ef476f' />
                    <circle cx='32' cy='35' r='17' fill='#f8f9fa' />
                    <circle cx='26' cy='33' r='2.5' fill='#1f2937' />
                    <circle cx='38' cy='33' r='2.5' fill='#1f2937' />
                    <path d='M32 36l-3 3h6z' fill='#ef476f' />
                    <path d='M25 41c2 2 4 3 7 3s5-1 7-3' stroke='#1f2937' strokeWidth='2' fill='none' strokeLinecap='round' />
                </svg>
            )
        case 'fox':
            return (
                <svg viewBox='0 0 64 64' style={iconCommonStyle(size)} aria-hidden='true'>
                    <rect width='64' height='64' rx='32' fill='#1f2937' />
                    <path d='M14 22l10 6 2-13zM50 22l-10 6-2-13z' fill='#f97316' />
                    <path d='M32 49c12 0 18-8 18-16 0-9-8-17-18-17s-18 8-18 17c0 8 6 16 18 16z' fill='#fb923c' />
                    <path d='M22 40c3 5 7 8 10 8s7-3 10-8l-10-6z' fill='#fff7ed' />
                    <circle cx='25.5' cy='32.5' r='2.3' fill='#111827' />
                    <circle cx='38.5' cy='32.5' r='2.3' fill='#111827' />
                </svg>
            )
        case 'owl':
            return (
                <svg viewBox='0 0 64 64' style={iconCommonStyle(size)} aria-hidden='true'>
                    <rect width='64' height='64' rx='32' fill='#0f172a' />
                    <ellipse cx='32' cy='36' rx='18' ry='16' fill='#8b5cf6' />
                    <circle cx='25' cy='33' r='7' fill='#f8fafc' />
                    <circle cx='39' cy='33' r='7' fill='#f8fafc' />
                    <circle cx='25' cy='33' r='3' fill='#0f172a' />
                    <circle cx='39' cy='33' r='3' fill='#0f172a' />
                    <path d='M32 36l-4 4h8z' fill='#f59e0b' />
                    <path d='M20 27l6-8 3 7zM44 27l-6-8-3 7z' fill='#7c3aed' />
                </svg>
            )
        case 'panda':
            return (
                <svg viewBox='0 0 64 64' style={iconCommonStyle(size)} aria-hidden='true'>
                    <rect width='64' height='64' rx='32' fill='#111827' />
                    <circle cx='21' cy='23' r='8' fill='#1f2937' />
                    <circle cx='43' cy='23' r='8' fill='#1f2937' />
                    <circle cx='32' cy='35' r='18' fill='#f9fafb' />
                    <ellipse cx='24.5' cy='34' rx='5.5' ry='7' fill='#1f2937' />
                    <ellipse cx='39.5' cy='34' rx='5.5' ry='7' fill='#1f2937' />
                    <circle cx='25' cy='34' r='2.2' fill='#f9fafb' />
                    <circle cx='39' cy='34' r='2.2' fill='#f9fafb' />
                    <path d='M32 37l-3 3h6z' fill='#111827' />
                </svg>
            )
        case 'koala':
            return (
                <svg viewBox='0 0 64 64' style={iconCommonStyle(size)} aria-hidden='true'>
                    <rect width='64' height='64' rx='32' fill='#334155' />
                    <circle cx='18' cy='28' r='10' fill='#94a3b8' />
                    <circle cx='46' cy='28' r='10' fill='#94a3b8' />
                    <circle cx='32' cy='36' r='17' fill='#cbd5e1' />
                    <circle cx='26' cy='34' r='2.5' fill='#1e293b' />
                    <circle cx='38' cy='34' r='2.5' fill='#1e293b' />
                    <ellipse cx='32' cy='40' rx='5.5' ry='4.5' fill='#64748b' />
                    <path d='M26 44c2 2 4 3 6 3s4-1 6-3' stroke='#1e293b' strokeWidth='2' fill='none' strokeLinecap='round' />
                </svg>
            )
        default:
            return null
    }
}
