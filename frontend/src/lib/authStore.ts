/**
 * Client-side auth state stored in localStorage.
 *
 * Active session:
 *   TOKEN_KEY  — JWT string
 *   USER_KEY   — JSON-serialised AuthUser
 *
 * Saved accounts (multi-profile):
 *   ACCOUNTS_KEY — JSON array of { token, user } for every signed-in account.
 *   Switching accounts swaps TOKEN_KEY/USER_KEY without removing others from the list.
 *   Sign Out removes the current account from the list and clears the active session.
 */

import type { AuthUser } from './api'
import {
	defaultAvatarForUser,
	type AvatarId,
} from './avatars'

const TOKEN_KEY = 'auth_token'
const USER_KEY = 'auth_user'
const ACCOUNTS_KEY = 'auth_accounts'

// ── Types ─────────────────────────────────────────────────────────────────────
export interface StoredAccount {
	token: string
	user: AuthUser
	avatar_id: AvatarId
}

function normalizeAccount(raw: Partial<StoredAccount>): StoredAccount | null {
	if (!raw.token || !raw.user) return null
	return {
		token: raw.token,
		user: raw.user,
		avatar_id: raw.avatar_id ?? defaultAvatarForUser(raw.user.id),
	}
}

function decodeJwtPayload(token: string): Record<string, unknown> | null {
	try {
		const parts = token.split('.')
		if (parts.length < 2) return null
		const payload = parts[1]
		const base64 = payload.replace(/-/g, '+').replace(/_/g, '/')
		const json = decodeURIComponent(
			atob(base64)
				.split('')
				.map(c => `%${(`00${c.charCodeAt(0).toString(16)}`).slice(-2)}`)
				.join(''),
		)
		return JSON.parse(json) as Record<string, unknown>
	} catch {
		return null
	}
}

function isTokenExpired(token: string): boolean {
	const payload = decodeJwtPayload(token)
	if (!payload) return true
	const exp = payload.exp
	if (typeof exp !== 'number') return true
	const nowSec = Math.floor(Date.now() / 1000)
	return exp <= nowSec
}

// ── Active session ─────────────────────────────────────────────────────────────
export function getToken(): string | null {
	if (typeof window === 'undefined') return null
	const token = localStorage.getItem(TOKEN_KEY)
	if (!token) return null
	if (isTokenExpired(token)) {
		clearAuth()
		return null
	}
	return token
}

export function getAuthUser(): AuthUser | null {
	if (typeof window === 'undefined') return null
	try {
		const raw = localStorage.getItem(USER_KEY)
		return raw ? (JSON.parse(raw) as AuthUser) : null
	} catch {
		return null
	}
}

export function isLoggedIn(): boolean {
	return !!getToken()
}

// ── Accounts list ─────────────────────────────────────────────────────────────
export function getAccounts(): StoredAccount[] {
	if (typeof window === 'undefined') return []
	try {
		const raw = localStorage.getItem(ACCOUNTS_KEY)
		if (!raw) return []
		const parsed = JSON.parse(raw) as Partial<StoredAccount>[]
		return parsed
			.map(normalizeAccount)
			.filter((a): a is StoredAccount => a !== null)
	} catch {
		return []
	}
}

function saveAccounts(accounts: StoredAccount[]): void {
	localStorage.setItem(ACCOUNTS_KEY, JSON.stringify(accounts))
}

// ── Mutations ──────────────────────────────────────────────────────────────────

/** Set active session and upsert into the saved accounts list. */
export function setAuth(token: string, user: AuthUser): void {
	localStorage.setItem(TOKEN_KEY, token)
	localStorage.setItem(USER_KEY, JSON.stringify(user))
	// Upsert into multi-account list
	const accounts = getAccounts()
	const idx = accounts.findIndex(a => a.user.id === user.id)
	if (idx >= 0) {
		accounts[idx] = {
			token,
			user,
			avatar_id: accounts[idx].avatar_id,
		}
	} else {
		accounts.push({
			token,
			user,
			avatar_id: defaultAvatarForUser(user.id),
		})
	}
	saveAccounts(accounts)
	window.dispatchEvent(new Event('auth-change'))
}

/** Switch to a saved account by userId (must already be in the accounts list). */
export function switchAccount(userId: number): void {
	const account = getAccounts().find(a => a.user.id === userId)
	if (!account) return
	localStorage.setItem(TOKEN_KEY, account.token)
	localStorage.setItem(USER_KEY, JSON.stringify(account.user))
	window.dispatchEvent(new Event('auth-change'))
}

export function getAvatarForUser(userId: number): AvatarId {
	const account = getAccounts().find(a => a.user.id === userId)
	if (!account) return defaultAvatarForUser(userId)
	return account.avatar_id
}

export function getCurrentAvatar(): AvatarId | null {
	const user = getAuthUser()
	if (!user) return null
	return getAvatarForUser(user.id)
}

export function setCurrentAvatar(avatarId: AvatarId): void {
	const current = getAuthUser()
	if (!current) return
	const accounts = getAccounts()
	const idx = accounts.findIndex(a => a.user.id === current.id)
	if (idx < 0) return
	accounts[idx] = { ...accounts[idx], avatar_id: avatarId }
	saveAccounts(accounts)
	window.dispatchEvent(new Event('auth-change'))
}

/**
 * Sign out of the current account.
 * Removes it from the saved list. If other accounts exist, switches to the
 * first one automatically; otherwise clears the active session entirely.
 */
export function clearAuth(): void {
	const current = getAuthUser()
	let accounts = getAccounts()
	if (current) {
		accounts = accounts.filter(a => a.user.id !== current.id)
		saveAccounts(accounts)
	}
	if (accounts.length > 0) {
		// Auto-switch to another saved account
		localStorage.setItem(TOKEN_KEY, accounts[0].token)
		localStorage.setItem(USER_KEY, JSON.stringify(accounts[0].user))
	} else {
		localStorage.removeItem(TOKEN_KEY)
		localStorage.removeItem(USER_KEY)
	}
	window.dispatchEvent(new Event('auth-change'))
}
