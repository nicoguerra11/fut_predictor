import { useState, useEffect, useCallback } from 'react'
import { WC_GROUPS, GROUP_LETTERS, getPairs } from '../data/groups'
import { TEAM_LOOKUP } from '../data/teams'
import styles from './GroupTable.module.css'

// ── Predicciones de un grupo ──────────────────────────────────────────────────

function useGroupPredictions(teams, apiBase, enabled) {
  const [pts, setPts] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!enabled) { setPts(null); return }
    setLoading(true)
    const pairs = getPairs(teams)
    Promise.all(
      pairs.map(([home, away]) =>
        fetch(`${apiBase}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            team_home: home,
            team_away: away,
            neutral: true,
            ranking_home: TEAM_LOOKUP[home]?.fifaRanking ?? null,
            ranking_away: TEAM_LOOKUP[away]?.fifaRanking ?? null,
          }),
        }).then((r) => r.json())
      )
    )
      .then((results) => {
        const acc = Object.fromEntries(teams.map((t) => [t, 0]))
        for (const r of results) {
          if (r.team_home in acc) acc[r.team_home] += r.prob_home * 3 + r.prob_draw
          if (r.team_away in acc) acc[r.team_away] += r.prob_away * 3 + r.prob_draw
        }
        setPts(acc)
      })
      .catch(() => setPts(null))
      .finally(() => setLoading(false))
  }, [enabled, teams.join(','), apiBase])

  return { pts, loading }
}

// ── Tarjeta de grupo ──────────────────────────────────────────────────────────

function GroupCard({ groupLetter, teams, apiBase, expanded }) {
  const { pts, loading } = useGroupPredictions(teams, apiBase, expanded)

  const sorted = [...teams].sort((a, b) => {
    if (!pts) return 0
    return (pts[b] ?? 0) - (pts[a] ?? 0)
  })

  return (
    <div className={`${styles.groupCard} ${expanded ? styles.groupCardExpanded : ''}`}>
      <div className={styles.groupHeader}>Grupo {groupLetter}</div>
      <table className={styles.table}>
        <thead>
          <tr>
            <th>Equipo</th>
            <th title="Puntos reales">PTS</th>
            <th title="Puntos predichos por el modelo">Pred.</th>
            <th title="Probabilidad estimada de clasificar">P(clasif.)</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((team, i) => {
            const teamPts = pts ? pts[team] : null
            const probAdv = pts ? estimateAdvance(sorted, team, pts) : null
            const info = TEAM_LOOKUP[team]

            return (
              <tr key={team} className={i < 2 && pts ? styles.qualifier : ''}>
                <td className={styles.teamCell}>
                  {info && <span className={styles.flag}>{info.flag}</span>}
                  <span>{team}</span>
                </td>
                <td>0</td>
                <td className={styles.predicted}>
                  {loading ? <span className={styles.shimmer} /> : teamPts !== null ? teamPts.toFixed(1) : '—'}
                </td>
                <td className={styles.probCol}>
                  {loading ? (
                    <span className={styles.shimmer} style={{ width: 48 }} />
                  ) : probAdv !== null ? (
                    <div className={styles.probCell}>
                      <div className={styles.probBar} style={{ width: `${(probAdv * 100).toFixed(0)}%` }} />
                      <span>{(probAdv * 100).toFixed(0)}%</span>
                    </div>
                  ) : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** Estima la probabilidad de clasificar de cada equipo a partir de los puntos predichos. */
function estimateAdvance(sortedTeams, team, pts) {
  const rank = sortedTeams.indexOf(team)
  const n = sortedTeams.length
  return Math.max(0, (n - rank) / n)
}

// ── Componente principal ──────────────────────────────────────────────────────

export default function GroupTable({ apiBase }) {
  const [selected, setSelected] = useState(null) // null = todos

  return (
    <div className={styles.container}>
      <h2 className={styles.title}>Tabla de grupos</h2>
      <p className={styles.subtitle}>
        Predicciones del modelo bayesiano para cada grupo. Seleccioná un grupo para ver los puntos esperados.
      </p>

      {/* ── Selector de grupo ──────────────────────────────────────── */}
      <div className={styles.selector}>
        <button
          className={`${styles.selectorBtn} ${selected === null ? styles.selectorBtnActive : ''}`}
          onClick={() => setSelected(null)}
        >
          Todos
        </button>
        {GROUP_LETTERS.map((letter) => (
          <button
            key={letter}
            className={`${styles.selectorBtn} ${selected === letter ? styles.selectorBtnActive : ''}`}
            onClick={() => setSelected((prev) => (prev === letter ? null : letter))}
          >
            {letter}
          </button>
        ))}
      </div>

      {/* ── Grid de grupos ─────────────────────────────────────────── */}
      <div className={`${styles.grid} ${selected ? styles.gridSingle : ''}`}>
        {(selected ? [selected] : GROUP_LETTERS).map((letter) => (
          <GroupCard
            key={letter}
            groupLetter={letter}
            teams={WC_GROUPS[letter]}
            apiBase={apiBase}
            expanded={selected === letter}
          />
        ))}
      </div>

      {!selected && (
        <p className={styles.hint}>
          Hacé clic en un grupo para ver los puntos predichos.
        </p>
      )}
    </div>
  )
}
