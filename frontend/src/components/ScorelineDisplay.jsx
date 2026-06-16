import { useState } from 'react'
import styles from './ScorelineDisplay.module.css'

const GRID_SIZE = 6

function cellStyle(prob, maxProb) {
  const ratio = maxProb > 0 ? prob / maxProb : 0
  if (ratio > 0.70) return { background: '#1a1a2e', color: '#e8c84a' }
  if (ratio > 0.45) return { background: '#2d3561', color: '#e8c84a' }
  if (ratio > 0.25) return { background: '#3d4f8a', color: 'white' }
  if (ratio > 0.12) return { background: '#b5d4f4', color: '#042C53' }
  return { background: '#e6f1fb', color: '#185FA5' }
}

function outcomeLabel(home, away) {
  if (home > away) return 'Local'
  if (home === away) return 'Empate'
  return 'Visitante'
}

export default function ScorelineDisplay({ scorelines, teamHome, teamAway }) {
  const [tooltip, setTooltip] = useState(null)

  if (!scorelines || scorelines.length === 0) return null

  // Construir lookup para el heatmap
  const lookup = {}
  for (const s of scorelines) {
    lookup[`${s.home}-${s.away}`] = s.prob
  }
  const maxProb = scorelines[0].prob
  const top6 = scorelines.slice(0, 6)

  return (
    <div className={styles.container}>
      {/* ── Heatmap 6×6 ─────────────────────────────────────────────── */}
      <h3 className={styles.sectionTitle}>Distribución de marcadores exactos</h3>

      <div className={styles.heatmapWrap}>
        {/* Etiqueta equipo visitante (eje X) */}
        <div className={styles.awayAxisLabel}>{teamAway} →</div>

        <div className={styles.heatmapInner}>
          {/* Encabezado de columnas (goles visitante 0–5) */}
          <div className={styles.colHeader}>
            <div className={styles.cornerCell} />
            {Array.from({ length: GRID_SIZE }, (_, j) => (
              <div key={j} className={styles.axisCell}>{j}</div>
            ))}
          </div>

          {/* Filas (goles local 0–5) con etiqueta del equipo local */}
          <div className={styles.gridRows}>
            <div className={styles.homeAxisLabel}>
              <span>{teamHome}</span>
            </div>
            <div className={styles.grid}>
              {Array.from({ length: GRID_SIZE }, (_, i) => (
                <div key={i} className={styles.row}>
                  <div className={styles.axisCell}>{i}</div>
                  {Array.from({ length: GRID_SIZE }, (_, j) => {
                    const prob = lookup[`${i}-${j}`] ?? 0
                    const isMax = prob === maxProb
                    const style = cellStyle(prob, maxProb)
                    const pct = (prob * 100).toFixed(1)
                    return (
                      <div
                        key={j}
                        className={`${styles.cell} ${isMax ? styles.cellMax : ''}`}
                        style={style}
                        onMouseEnter={() => setTooltip({ i, j, pct })}
                        onMouseLeave={() => setTooltip(null)}
                      >
                        {pct}%
                        {tooltip && tooltip.i === i && tooltip.j === j && (
                          <div className={styles.tooltip}>
                            {i} – {j}: {pct}%
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ── Top 6 marcadores ─────────────────────────────────────────── */}
      <h3 className={styles.sectionTitle} style={{ marginTop: '1.5rem' }}>
        Marcadores más probables
      </h3>

      <div className={styles.rankList}>
        {top6.map((s, idx) => {
          const barWidth = maxProb > 0 ? (s.prob / maxProb) * 100 : 0
          const isFirst = idx === 0
          return (
            <div key={`${s.home}-${s.away}`} className={styles.rankRow}>
              <div className={`${styles.rankPos} ${isFirst ? styles.rankFirst : ''}`}>
                {idx + 1}
              </div>
              <div className={styles.scoreBadge}>
                {s.home} – {s.away}
              </div>
              <div className={styles.outcomeTag}>
                {outcomeLabel(s.home, s.away)}
              </div>
              <div className={styles.barTrack}>
                <div
                  className={styles.rankBar}
                  style={{ width: `${barWidth.toFixed(1)}%` }}
                />
              </div>
              <div className={styles.rankPct}>
                {(s.prob * 100).toFixed(1)}%
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
