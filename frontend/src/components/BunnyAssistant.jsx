import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useRive, useStateMachineInput } from '@rive-app/react-canvas';
import { AnimatePresence, motion } from 'framer-motion';

// ──────────────────────────────────────────────────────────────
//  Rive 상태머신 이름과 입력 이름은 .riv 파일에 따라 조정하세요.
//  아래는 'interative-bunny.charater.riv'에서 가장 일반적으로
//  사용되는 관례 이름입니다.
// ──────────────────────────────────────────────────────────────
const RIVE_STATE_MACHINE = 'State Machine 1';
const INPUT_LISTENING   = 'isListening';   // Boolean input
const INPUT_THINKING    = 'isThinking';    // Boolean input
const INPUT_TALKING     = 'isTalking';     // Boolean input
const RIVE_SOURCE_FILE  = '/gbunny.riv';

// ──────────────────────────────────────────────────────────────
//  OpenAI 클라이언트 (VITE_OPENAI_API_KEY 환경 변수 사용)
//  .env 파일에: VITE_OPENAI_API_KEY=sk-...
// ──────────────────────────────────────────────────────────────
function createOpenAIClient() {
  return { ready: true };
}

// ──────────────────────────────────────────────────────────────
//  캐릭터 상태 타입
// ──────────────────────────────────────────────────────────────
const CHAR_STATE = {
  IDLE:      'idle',
  LISTENING: 'listening',
  THINKING:  'thinking',
  TALKING:   'talking',
};

// ──────────────────────────────────────────────────────────────
//  상태 레이블 (UI 배지)
// ──────────────────────────────────────────────────────────────
const STATE_LABEL = {
  [CHAR_STATE.IDLE]:      { text: 'Idle',       color: 'bg-white/20 text-white/60' },
  [CHAR_STATE.LISTENING]: { text: 'Listening…', color: 'bg-indigo-500/40 text-indigo-200' },
  [CHAR_STATE.THINKING]:  { text: 'Thinking…',  color: 'bg-amber-500/40 text-amber-200' },
  [CHAR_STATE.TALKING]:   { text: 'Talking…',   color: 'bg-emerald-500/40 text-emerald-200' },
};

// ──────────────────────────────────────────────────────────────
//  말풍선 (AI 응답 텍스트 타이핑 효과)
// ──────────────────────────────────────────────────────────────
function TypingText({ text }) {
  const [displayed, setDisplayed] = useState('');

  useEffect(() => {
    setDisplayed('');
    if (!text) return;
    let i = 0;
    const id = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) clearInterval(id);
    }, 18);
    return () => clearInterval(id);
  }, [text]);

  return <span>{displayed}</span>;
}

// ──────────────────────────────────────────────────────────────
//  채팅 말풍선
// ──────────────────────────────────────────────────────────────
function ChatBubble({ role, content, isLatest }) {
  const isUser = role === 'user';
  return (
    <motion.div
      initial={{ opacity: 0, y: 8, scale: 0.97 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.22 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-2`}
    >
      <div
        className={`max-w-[78%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed backdrop-blur-sm shadow-lg
          ${isUser
            ? 'bg-indigo-500/60 text-white rounded-br-sm'
            : 'bg-white/10 text-white/90 rounded-bl-sm border border-white/10'
          }`}
      >
        {!isUser && isLatest ? <TypingText text={content} /> : content}
      </div>
    </motion.div>
  );
}

// ──────────────────────────────────────────────────────────────
//  메인 컴포넌트
// ──────────────────────────────────────────────────────────────
export default function BunnyAssistant() {
  const [messages, setMessages]         = useState([]);
  const [inputText, setInputText]       = useState('');
  const [charState, setCharState]       = useState(CHAR_STATE.IDLE);
  const [isLoading, setIsLoading]       = useState(false);
  const [error, setError]               = useState(null);
  const [riveLoadFailed, setRiveLoadFailed] = useState(false);
  const chatEndRef                      = useRef(null);
  const inputRef                        = useRef(null);
  const openaiRef                       = useRef(null);

  // OpenAI 클라이언트 초기화
  useEffect(() => {
    openaiRef.current = createOpenAIClient();
  }, []);

  // ── Rive 설정 ──────────────────────────────────────────────
  const { RiveComponent, rive } = useRive({
    src: RIVE_SOURCE_FILE,
    stateMachines: RIVE_STATE_MACHINE,
    autoplay: true,
    onLoadError: (err) => {
      setRiveLoadFailed(true);
      console.warn('Rive load error:', err);
    },
  });

  const inputListening = useStateMachineInput(rive, RIVE_STATE_MACHINE, INPUT_LISTENING);
  const inputThinking  = useStateMachineInput(rive, RIVE_STATE_MACHINE, INPUT_THINKING);
  const inputTalking   = useStateMachineInput(rive, RIVE_STATE_MACHINE, INPUT_TALKING);

  const setRiveState = useCallback((state) => {
    if (!rive) return;
    // 모든 boolean 입력을 false로 리셋
    [inputListening, inputThinking, inputTalking].forEach((inp) => {
      if (inp) inp.value = false;
    });
    // 해당 상태만 켜기
    if (state === CHAR_STATE.LISTENING && inputListening) inputListening.value = true;
    if (state === CHAR_STATE.THINKING  && inputThinking)  inputThinking.value  = true;
    if (state === CHAR_STATE.TALKING   && inputTalking)   inputTalking.value   = true;
  }, [rive, inputListening, inputThinking, inputTalking]);

  // ── charState 변경 시 Rive 동기화 ─────────────────────────
  useEffect(() => {
    setRiveState(charState);
  }, [charState, setRiveState]);

  // ── 자동 스크롤 ────────────────────────────────────────────
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // ── 메시지 전송 ────────────────────────────────────────────
  const handleSend = useCallback(async () => {
    const text = inputText.trim();
    if (!text || isLoading) return;

    setError(null);
    setInputText('');
    setIsLoading(true);

    const userMsg = { role: 'user', content: text };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);

    // 1) Listening 상태
    setCharState(CHAR_STATE.LISTENING);
    await new Promise((r) => setTimeout(r, 600));

    // 2) Thinking 상태 (API 호출)
    setCharState(CHAR_STATE.THINKING);

    try {
      if (!openaiRef.current?.ready) {
        throw new Error('OpenAI 클라이언트를 초기화하지 못했습니다.');
      }

      const response = await fetch('/api/chat/openai', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [
            {
              role: 'system',
              content: '당신은 친절하고 귀여운 AI 어시스턴트입니다. 간결하고 자연스럽게 답변하세요.',
            },
            ...newMessages,
          ],
          temperature: 0.7,
          max_tokens: 500,
        }),
      });

      const payload = await response.json();
      if (!response.ok || payload?.status !== 'ok') {
        const message = String(payload?.detail || payload?.error?.message || 'OpenAI 호출 실패');
        throw new Error(message);
      }

      const assistantContent = payload?.content || '(응답 없음)';
      const assistantMsg = { role: 'assistant', content: assistantContent };

      // 3) Talking 상태 (타이핑 표시)
      setCharState(CHAR_STATE.TALKING);
      setMessages([...newMessages, assistantMsg]);

      // 타이핑 시간 ≈ 글자 수 × 18ms (최소 1.5s)
      const talkDuration = Math.max(1500, assistantContent.length * 18);
      await new Promise((r) => setTimeout(r, talkDuration));

    } catch (err) {
      console.error(err);
      setError(err.message || '오류가 발생했습니다.');
      setMessages([...newMessages, { role: 'assistant', content: `⚠️ ${err.message}` }]);
      setCharState(CHAR_STATE.TALKING);
      await new Promise((r) => setTimeout(r, 1200));
    } finally {
      // 4) Idle 복귀
      setCharState(CHAR_STATE.IDLE);
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [inputText, isLoading, messages]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const stateLabel = STATE_LABEL[charState];

  // ── 캐릭터 글로우 색상 ──────────────────────────────────────
  const glowColor = {
    [CHAR_STATE.IDLE]:      'rgba(148,130,255,0.25)',
    [CHAR_STATE.LISTENING]: 'rgba(99,102,241,0.45)',
    [CHAR_STATE.THINKING]:  'rgba(245,158,11,0.45)',
    [CHAR_STATE.TALKING]:   'rgba(16,185,129,0.45)',
  }[charState];

  return (
    <div
      className="min-h-screen w-full flex flex-col items-center justify-between overflow-hidden"
      style={{
        background: 'radial-gradient(ellipse at 60% 10%, #1a1040 0%, #0d0d1a 60%, #000 100%)',
        fontFamily: "'Toss Product Sans', 'Pretendard', 'Apple SD Gothic Neo', 'Noto Sans KR', system-ui, sans-serif",
      }}
    >
      {/* ── 배경 장식 ── */}
      <div
        className="pointer-events-none fixed inset-0"
        style={{
          background:
            'radial-gradient(circle at 20% 80%, rgba(99,102,241,0.08) 0%, transparent 50%), ' +
            'radial-gradient(circle at 80% 20%, rgba(168,85,247,0.08) 0%, transparent 50%)',
        }}
      />

      {/* ── 헤더 ── */}
      <header className="w-full flex items-center justify-between px-6 py-4 z-10">
        <div className="flex items-center gap-2">
          <span className="text-white/80 font-semibold tracking-wide text-sm">🐰 Bunny AI</span>
        </div>
        <motion.span
          key={charState}
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.2 }}
          className={`text-xs font-medium px-3 py-1 rounded-full backdrop-blur-sm ${stateLabel.color}`}
        >
          {stateLabel.text}
        </motion.span>
      </header>

      {/* ── 캐릭터 영역 ── */}
      <main className="flex-1 flex flex-col items-center justify-center w-full z-10">
        <motion.div
          animate={{
            boxShadow: `0 0 60px 20px ${glowColor}`,
          }}
          transition={{ duration: 0.6, ease: 'easeInOut' }}
          className="rounded-full overflow-hidden"
          style={{ width: 280, height: 280 }}
        >
          {!riveLoadFailed ? (
            <RiveComponent
              style={{ width: '100%', height: '100%' }}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-8xl bg-white/5 text-white/70">
              🐰
            </div>
          )}
        </motion.div>

        {/* 상태 펄스 링 */}
        <AnimatePresence>
          {charState !== CHAR_STATE.IDLE && (
            <motion.div
              key="pulse-ring"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: [0.6, 0], scale: [1, 1.6] }}
              exit={{ opacity: 0 }}
              transition={{ duration: 1.2, repeat: Infinity }}
              className="absolute rounded-full pointer-events-none"
              style={{
                width: 280,
                height: 280,
                border: `2px solid ${glowColor}`,
              }}
            />
          )}
        </AnimatePresence>
      </main>

      {/* ── 대화 패널 ── */}
      <section
        className="w-full max-w-2xl z-10 px-4 pb-4 flex flex-col gap-3"
        style={{ maxHeight: '45vh' }}
      >
        {/* 채팅 히스토리 */}
        <div
          className="flex-1 overflow-y-auto pr-1 scrollbar-thin"
          style={{ maxHeight: 'calc(45vh - 80px)', minHeight: 60 }}
        >
          <AnimatePresence initial={false}>
            {messages.map((msg, i) => (
              <ChatBubble
                key={i}
                role={msg.role}
                content={msg.content}
                isLatest={i === messages.length - 1 && msg.role === 'assistant'}
              />
            ))}
          </AnimatePresence>
          <div ref={chatEndRef} />
        </div>

        {/* 입력창 */}
        <motion.div
          className="flex items-end gap-2 rounded-2xl p-1 backdrop-blur-xl"
          style={{
            background: 'rgba(255,255,255,0.07)',
            border: '1px solid rgba(255,255,255,0.12)',
            boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
          }}
          whileFocus={{ scale: 1.005 }}
        >
          <textarea
            ref={inputRef}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="메시지를 입력하세요… (Enter로 전송)"
            rows={1}
            disabled={isLoading}
            className="flex-1 bg-transparent text-white/90 placeholder-white/30 text-sm resize-none outline-none px-3 py-2.5"
            style={{ maxHeight: 120, lineHeight: '1.5' }}
            onInput={(e) => {
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
            }}
          />
          <motion.button
            onClick={handleSend}
            disabled={isLoading || !inputText.trim()}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="shrink-0 w-10 h-10 rounded-xl flex items-center justify-center text-white font-bold
              disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            style={{
              background: isLoading
                ? 'rgba(99,102,241,0.3)'
                : 'linear-gradient(135deg,#6366f1,#8b5cf6)',
              boxShadow: isLoading ? 'none' : '0 4px 16px rgba(99,102,241,0.5)',
            }}
          >
            {isLoading ? (
              <motion.span
                animate={{ rotate: 360 }}
                transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}
              >
                ⟳
              </motion.span>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
              </svg>
            )}
          </motion.button>
        </motion.div>

        {/* API 키 미설정 안내 */}
        <p className="text-center text-xs text-amber-400/70">
          OPENAI_API_KEY 는 백엔드 환경변수에 설정되어 있어야 합니다.
        </p>
      </section>
    </div>
  );
}
