'use client'

import { useEffect, useState } from 'react'
import { getLang, setLang as _setLang, t, type Lang, type Translations } from './i18n'

export function useTranslation(): {
  lang: Lang
  tr: Translations
  setLang: (lang: Lang) => void
} {
  const [lang, setLangState] = useState<Lang>('en')

  // Hydrate from localStorage on mount (avoids SSR mismatch)
  useEffect(() => {
    setLangState(getLang())
    const handler = () => setLangState(getLang())
    window.addEventListener('lang-change', handler)
    return () => window.removeEventListener('lang-change', handler)
  }, [])

  return {
    lang,
    tr: t(lang),
    setLang: (l: Lang) => {
      _setLang(l)
      setLangState(l)
    },
  }
}
