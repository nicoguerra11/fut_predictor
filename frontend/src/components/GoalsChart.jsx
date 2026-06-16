import styles from './GoalsChart.module.css'

/**
 * Gráfico de barras de la distribución de goles (0–8) para un equipo.
 * Resalta el intervalo de credibilidad 89%.
 */
export default function GoalsChart({ dist, label, ci, color }) {
  const maxProb = Math.max(...dist)

  return (
    <div className={styles.container}>
      <div className={styles.label}>{label}</div>
      <div className={styles.bars}>
        {dist.map((prob, goals) => {
          const inCI = goals >= ci[0] && goals <= ci[1]
          const height = maxProb > 0 ? (prob / maxProb) * 100 : 0
          return (
            <div key={goals} className={styles.col}>
              <div className={styles.barWrap}>
                <div
                  className={`${styles.bar} ${inCI ? styles.inCI : ''}`}
                  style={{
                    height: `${height}%`,
                    background: inCI ? color : `${color}55`,
                  }}
                  title={`${goals} goles: ${(prob * 100).toFixed(1)}%`}
                />
              </div>
              <div className={styles.goalLabel}>{goals}</div>
            </div>
          )
        })}
      </div>
      <div className={styles.ciNote} style={{ color }}>
        IC 89%: {ci[0]}–{ci[1]} goles
      </div>
    </div>
  )
}
