import React, { useEffect, useState, useRef } from 'react'
import axios from 'axios'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer
} from 'recharts'

const API_BASE = process.env.REACT_APP_API || 'http://localhost:8000'

export default function App() {
  const [status, setStatus] = useState({ running: false, file_count: 0, vector_count: 0, total_time: 0 })
  const [results, setResults] = useState([])
  const [frames, setFrames] = useState([])
  const [frameIndex, setFrameIndex] = useState(0)
  const [series, setSeries] = useState({}) // per-product time series
  const animRef = useRef(null)
  const [chatInput, setChatInput] = useState('')
  const [chatMessages, setChatMessages] = useState([])
  const [live, setLive] = useState(false)

  useEffect(() => {
    fetchStatus()
    loadResults()
  }, [])

  // live 모드일 때 주기적으로 결과 갱신
  useEffect(() => {
    let id = null
    if (live) {
      id = setInterval(() => {
        loadResults()
        fetchStatus()
      }, 3000)
    }
    return () => { if (id) clearInterval(id) }
  }, [live])

  async function fetchStatus() {
    try {
      const r = await axios.get(`${API_BASE}/status`)
      setStatus(r.data)
      const vc = await axios.get(`${API_BASE}/vector-count`)
      setStatus(prev => ({ ...prev, vector_count: vc.data.count }))
    } catch (e) {
      console.error(e)
    }
  }

  async function startAnalysis() {
    try {
      await axios.post(`${API_BASE}/run-analysis`)
      setStatus(prev => ({ ...prev, running: true }))
      // poll for results
      const id = setInterval(async () => {
        const s = await axios.get(`${API_BASE}/status`)
        setStatus(s.data)
        if (!s.data.running) {
          clearInterval(id)
          loadResults()
        }
      }, 1000)
    } catch (e) {
      console.error(e)
    }
  }

  async function loadResults() {
    try {
      const r = await axios.get(`${API_BASE}/results`)
      const data = r.data.results || []
      setResults(data)

      // build per-product latest points and extend series for animation
      const products = ["C6","C9","C11","C12"]
      const colors = { C6: '#8884d8', C9: '#82ca9d', C11: '#ffc658', C12: '#ff7f7f' }

      setSeries(prev => {
        const next = { ...prev }
        const ts = Date.now()
        products.forEach(p => {
          const hits = data.filter(d => (d.product || '') === p)
          const latest = hits.length > 0 ? hits[hits.length-1] : null
          const score = latest?.risk?.score ?? latest?.score ?? 0
          if (!next[p]) next[p] = []
          // append new point
          next[p] = [...next[p].slice(-40), { time: ts, score, grade: latest?.risk?.grade ?? latest?.grade ?? 'N/A' }]
        })
        return next
      })

      // construct frames similar to original app: gradual score scaling
      const riskData = data.map((rObj, i) => {
        // backend returns { product, in_fields, out_fields, in_mapping, out_mapping, risk }
        const risk = (rObj && rObj.risk) ? rObj.risk : (rObj || {})
        const score = risk?.score ?? rObj.score ?? 0
        const grade = risk?.grade ?? rObj.grade ?? 'N/A'
        const financial = risk?.details?.financial ?? rObj.details?.financial ?? rObj.financial ?? 0
        const credit = risk?.details?.credit ?? rObj.details?.credit ?? rObj.credit ?? 0
        const behavior = risk?.details?.behavior ?? rObj.details?.behavior ?? rObj.behavior ?? 0
        const regulation = risk?.details?.regulation ?? rObj.details?.regulation ?? rObj.regulation ?? 0
        return { index: i, score, grade, financial, credit, behavior, regulation }
      })

      // create simple frame list for legacy score line (keep for compatibility)
      const fframes = []
      for (let t = 1; t <= Math.max(1, riskData.length); t++) {
        const temp = riskData.map(d => ({ ...d }))
        temp.forEach(d => { d.time = t; d.score = d.score * (t / Math.max(1, riskData.length)) })
        fframes.push(...temp)
      }
      setFrames(fframes)
      setFrameIndex(0)

      // start animation for legacy frames
      if (animRef.current) clearInterval(animRef.current)
      animRef.current = setInterval(() => {
        setFrameIndex(i => {
          const next = i + 1
          if (next >= fframes.length) {
            clearInterval(animRef.current)
            return fframes.length - 1
          }
          return next
        })
      }, 150)

    } catch (e) { console.error(e) }
  }

  // manual refresh for status and results
  async function refreshAll() {
    await fetchStatus()
    await loadResults()
  }

  async function sendChat() {
    if (!chatInput) return
    setChatMessages(prev => [...prev, { role: 'user', text: chatInput }])
    try {
      const r = await axios.post(`${API_BASE}/chat`, { question: chatInput })
      const ans = r.data.answer || ''
      setChatMessages(prev => [...prev, { role: 'ai', text: ans }])
      setChatInput('')
    } catch (e) {
      console.error(e)
      setChatMessages(prev => [...prev, { role: 'ai', text: '서버에 연결할 수 없습니다.' }])
    }
  }

  const currentFrame = frames[frameIndex] ? frames[frameIndex] : null

  return (
    <div className="app">
      <div className="side">
        <div className="header">AI 대출 심사 컨트롤</div>
        <div style={{marginBottom:12}}>
          <button className="button" onClick={startAnalysis}>전체 분석 실행</button>
          <button className="button" style={{marginLeft:8, background:'#6c757d'}} onClick={refreshAll}>새로고침</button>
        </div>
        <div className="metrics">
          <div className="metric">
            <div>파일 수</div>
            <div style={{fontSize:20, fontWeight:700}}>{status.file_count}</div>
          </div>
          <div className="metric">
            <div>벡터 수</div>
            <div style={{fontSize:20, fontWeight:700}}>{status.vector_count}</div>
          </div>
        </div>
        <div style={{marginTop:12}}>
          <div style={{fontSize:14, color:'#666'}}>상태: {status.running ? '분석 중...' : '준비'}</div>
          <div style={{fontSize:12, color:'#999'}}>총 소요: {Math.round(status.total_time*100)/100}s</div>
        </div>

        <div style={{marginTop:18}}>
          <div style={{fontWeight:700, marginBottom:8}}>채팅 (AI 전략)</div>
          <div style={{display:'flex', flexDirection:'column', gap:8}}>
            <div style={{maxHeight:200, overflowY:'auto', background:'#fff', padding:8, borderRadius:6}}>
              {chatMessages.map((m, idx) => (
                <div key={idx} style={{marginBottom:6, textAlign: m.role==='user' ? 'right' : 'left'}}>
                  <div style={{display:'inline-block', padding:'6px 10px', borderRadius:8, background: m.role==='user' ? '#0b74de' : '#f1f3f5', color: m.role==='user' ? '#fff' : '#222'}}>{m.text}</div>
                </div>
              ))}
            </div>
            <div style={{display:'flex', gap:8}}>
              <input value={chatInput} onChange={e=>setChatInput(e.target.value)} placeholder="질문을 입력하세요" style={{flex:1, padding:8, borderRadius:6, border:'1px solid #ddd'}} />
              <button className="button" onClick={sendChat}>전송</button>
            </div>
          </div>
        </div>
      </div>

      <div className="main">
        <h2>실시간 리스크 & 벡터 분석</h2>

        <div style={{height:320}}>
          <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:8}}>
            <h4 style={{margin:0}}>제품별 실시간 리스크 (Score)</h4>
            <div>
              <label style={{marginRight:8}}>라이브: </label>
              <input type="checkbox" checked={live} onChange={e=>setLive(e.target.checked)} />
            </div>
          </div>
          <ResponsiveContainer width="100%" height={260}>
            {/* use full timeline from first product as chart data fallback */}
            <LineChart data={series[products[0]] || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tickFormatter={(t)=>new Date(t).toLocaleTimeString()} />
              <YAxis />
              <Tooltip labelFormatter={(t)=>new Date(t).toLocaleString()} />
              <Legend />
              {/* draw a line per product */}
              {products.map((p, idx) => (
                <Line key={p} data={series[p] || []} dataKey="score" name={p} stroke={{C6:'#8884d8',C9:'#82ca9d',C11:'#ffc658',C12:'#ff7f7f'}[p] || '#666'} dot={false} isAnimationActive={true} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div style={{display:'flex', gap:12, marginTop:12}}>
          {/* Grade badges per product */}
          {['C6','C9','C11','C12'].map(p => {
            const last = (series[p] || []).slice(-1)[0]
            const grade = last?.grade || 'N/A'
            const score = last?.score ?? 0
            const color = grade === 'HIGH' ? '#ff4d4f' : grade === 'MEDIUM' ? '#ffa940' : '#52c41a'
            return (
              <div key={p} style={{flex:1, background:'#fff', padding:12, borderRadius:8, boxShadow:'0 1px 4px rgba(0,0,0,0.06)'}}>
                <div style={{fontWeight:700}}>{p}</div>
                <div style={{fontSize:20, fontWeight:800, color:color}}>{Math.round(score)}</div>
                <div style={{marginTop:6, fontSize:13}}>등급: <span style={{fontWeight:700}}>{grade}</span></div>
              </div>
            )
          })}
        </div>

        <div style={{display:'flex', gap:20, marginTop:20}}>
          <div style={{flex:2, height:300}}>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={results.map((r,i)=>({ index:i, financial: r.details?.financial ?? r.financial ?? 0, credit: r.details?.credit ?? r.credit ?? 0, behavior: r.details?.behavior ?? r.behavior ?? 0, regulation: r.details?.regulation ?? r.regulation ?? 0 }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="index" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="financial" stackId="a" fill="#8884d8" />
                <Bar dataKey="credit" stackId="a" fill="#82ca9d" />
                <Bar dataKey="behavior" stackId="a" fill="#ffc658" />
                <Bar dataKey="regulation" stackId="a" fill="#ff7f7f" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{flex:1, height:300}}>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie data={results.map((r,i)=>({ name: r.grade ?? 'N', value: r.score ?? 0 }))} dataKey="value" nameKey="name" outerRadius={80} fill="#8884d8">
                  {results.map((_, idx) => <Cell key={idx} fill={['#8884d8', '#82ca9d', '#ffc658', '#ff7f7f'][idx % 4]} />)}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={{marginTop:20}}>
          <h3>분석 결과</h3>
          <div style={{maxHeight:300, overflowY:'auto', background:'#fff', padding:12, borderRadius:8}}>
            {results.length === 0 && <div style={{color:'#666'}}>결과가 없습니다.</div>}
            {results.map((r, idx) => (
              <div key={idx} style={{padding:8, borderBottom:'1px solid #f0f0f0'}}>
                <div style={{fontWeight:700}}>#{idx} {r.product ?? ''} — 점수: {r.risk?.score ?? r.score ?? 'N/A'} 등급: {r.risk?.grade ?? r.grade ?? 'N/A'}</div>
                <div style={{fontSize:13, color:'#444', marginTop:6}}>금융: {r.risk?.details?.financial ?? r.details?.financial ?? 0} | 신용: {r.risk?.details?.credit ?? r.details?.credit ?? 0} | 행태: {r.risk?.details?.behavior ?? r.details?.behavior ?? 0} | 규제: {r.risk?.details?.regulation ?? r.details?.regulation ?? 0}</div>
                <details style={{marginTop:8}}>
                  <summary>원본/매핑 보기</summary>
                  <pre style={{whiteSpace:'pre-wrap', fontSize:12}}>{JSON.stringify(r, null, 2)}</pre>
                </details>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  )
}
