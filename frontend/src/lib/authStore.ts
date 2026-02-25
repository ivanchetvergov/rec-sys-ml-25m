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

const TOKEN_KEY = 'auth_token'
const USER_KEY = 'auth_user'
const ACCOUNTS_KEY = 'auth_accounts'

// ── Types ─────────────────────────────────────────────────────────────────────
export interface StoredAccount {
	token: string
	user: AuthUser
}

// ── Active session ─────────────────────────────────────────────────────────────
export function getToken(): string | null {
	if (typeof window === 'undefined') return null
	return localStorage.getItem(TOKEN_KEY)
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
		return raw ? (JSON.parse(raw) as StoredAccount[]) : []
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
		accounts[idx] = { token, user }
	} else {
		accounts.push({ token, user })
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
