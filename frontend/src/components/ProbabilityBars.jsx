import { useEffect, useRef } from 'react'
import styles from './ProbabilityBars.module.css'

export default function ProbabilityBars({ probHome, probDraw, probAway, labelHome, labelAway }) {
  const homeRef = useRef(null)
  const drawRef = useRef(null)
  const awayRef = useRef(null)

  useEffect(() => {
    // Animar barras desde 0 al valor real
    const delay = (ms) => new Promise((r) => setTimeout(r, ms))
    async function animate() {
      await delay(50)
      if (homeRef.current) homeRef.current.style.width = `${(probHome * 100).toFixed(1)}%`
      if (drawRef.current) drawRef.current.style.width = `${(probDraw * 100).toFixed(1)}%`
      if (awayRef.current) awayRef.current.style.width = `${(probAway * 100).toFixed(1)}%`
    }
    animate()
  }, [probHome, probDraw, probAway])

  const bars = [
    { label: labelHome, prob: probHome, ref: homeRef, colorClass: styles.barHome },
    { label: 'Empate', prob: probDraw, ref: drawRef, colorClass: styles.barDraw },
    { label: labelAway, prob: probAway, ref: awayRef, colorClass: styles.barAway },
  ]

  return (
    <div className={styles.container}>
      <h3 className={styles.heading}>Probabilidades de resultado</h3>
      {bars.map(({ label, prob, ref, colorClass }) => (
        <div key={label} className={styles.row}>
          <span className={styles.rowLabel}>{label}</span>
          <div className={styles.track}>
            <div
              ref={ref}
              className={`${styles.bar} ${colorClass}`}
              style={{ width: '0%' }}
            />
          </div>
          <span className={styles.pct}>{(prob * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}
