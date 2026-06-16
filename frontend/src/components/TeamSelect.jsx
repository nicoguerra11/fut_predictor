import { useState, useRef, useEffect } from 'react'
import styles from './TeamSelect.module.css'

export default function TeamSelect({ teams, value, onChange, disabled = false }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [highlighted, setHighlighted] = useState(-1)
  const containerRef = useRef(null)
  const inputRef = useRef(null)
  const listRef = useRef(null)

  const selected = teams.find((t) => t.name === value) || null
  const filtered = query
    ? teams.filter((t) => t.name.toLowerCase().includes(query.toLowerCase()))
    : teams

  useEffect(() => {
    function onOutside(e) {
      if (containerRef.current && !containerRef.current.contains(e.target)) close()
    }
    document.addEventListener('mousedown', onOutside)
    return () => document.removeEventListener('mousedown', onOutside)
  }, [])

  useEffect(() => { setHighlighted(-1) }, [query])

  useEffect(() => {
    if (highlighted >= 0 && listRef.current) {
      const items = listRef.current.querySelectorAll('[data-opt]')
      items[highlighted]?.scrollIntoView({ block: 'nearest' })
    }
  }, [highlighted])

  function close() {
    setOpen(false)
    setQuery('')
    setHighlighted(-1)
  }

  function openDropdown() {
    if (disabled) return
    setOpen(true)
    setTimeout(() => inputRef.current?.focus(), 0)
  }

  function handleTriggerClick() {
    open ? close() : openDropdown()
  }

  function handleSelect(team) {
    onChange(team.name)
    close()
  }

  function handleKeyDown(e) {
    if (!open) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlighted((h) => Math.min(h + 1, filtered.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlighted((h) => Math.max(h - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      if (highlighted >= 0 && filtered[highlighted]) handleSelect(filtered[highlighted])
    } else if (e.key === 'Escape') {
      close()
    }
  }

  return (
    <div className={styles.root} ref={containerRef}>
      {/* ── Trigger ── */}
      <div
        className={`${styles.trigger} ${open ? styles.triggerOpen : ''} ${disabled ? styles.triggerDisabled : ''}`}
        onClick={handleTriggerClick}
        role="combobox"
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        {open ? (
          <input
            ref={inputRef}
            className={styles.searchInput}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Buscar selección..."
            autoComplete="off"
          />
        ) : (
          <span className={selected ? styles.selectedLabel : styles.placeholder}>
            {selected ? (
              <>
                <span className={styles.selectedFlag}>{selected.flag}</span>
                <span className={styles.selectedName}>{selected.name}</span>
              </>
            ) : (
              'Seleccioná un equipo'
            )}
          </span>
        )}
        <svg
          className={`${styles.chevron} ${open ? styles.chevronUp : ''}`}
          width="15" height="15" viewBox="0 0 24 24"
          fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </div>

      {/* ── Dropdown ── */}
      {open && (
        <div className={styles.dropdown} ref={listRef} role="listbox">
          {filtered.length === 0 ? (
            <div className={styles.empty}>Sin resultados para "{query}"</div>
          ) : (
            filtered.map((team, i) => (
              <div
                key={team.name}
                data-opt
                role="option"
                aria-selected={team.name === value}
                className={[
                  styles.option,
                  team.name === value ? styles.optionSelected : '',
                  i === highlighted ? styles.optionHighlighted : '',
                ].join(' ')}
                onMouseDown={() => handleSelect(team)}
                onMouseEnter={() => setHighlighted(i)}
              >
                <span className={styles.optionName}>{team.name}</span>
                <span className={styles.optionRank}>#{team.fifaRanking}</span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
