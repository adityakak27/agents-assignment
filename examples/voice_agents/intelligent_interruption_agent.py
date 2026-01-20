"""
Intelligent Interruption Handling Agent

This agent demonstrates context-aware interruption handling that:
- IGNORES backchannels (yeah, ok, hmm, uh-huh) while speaking
- INTERRUPTS on explicit commands (stop, wait, no) while speaking
- RESPONDS normally to any input when silent

Uses:
- LLM: Gemini 2.0 Flash (fast, capable)
- STT: Deepgram Nova-3 (speech-to-text)
- TTS: Deepgram Aura-2 (text-to-speech)

Required: Set DEEPGRAM_API_KEY (for STT and TTS) and GOOGLE_API_KEY (for LLM) environment variables

Usage:
    python intelligent_interruption_agent.py console  # Test locally with mic/speaker
    python intelligent_interruption_agent.py dev      # Run in development mode
"""

import logging
import sys
import time
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _add_to_syspath(path: Path) -> None:
    if not path.is_dir():
        return
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_add_to_syspath(ROOT / "livekit-agents")
plugins_root = ROOT / "livekit-plugins"
if plugins_root.is_dir():
    for plugin_dir in plugins_root.iterdir():
        if plugin_dir.is_dir() and (plugin_dir / "livekit").is_dir():
            _add_to_syspath(plugin_dir)

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    UserInputTranscribedEvent,
    cli,
    metrics,
    room_io,
)
from livekit.agents.voice.events import AgentStateChangedEvent, ConversationItemAddedEvent, SpeechCreatedEvent
from livekit.agents.llm.chat_context import ChatContext, ChatMessage
from livekit.agents.llm import function_tool
from livekit.plugins import silero, deepgram
from livekit.plugins.google import LLM as GeminiLLM
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from interruption_classifier import (
    InterruptionClassifier,
    InterruptionClassifierConfig,
    InterruptionIntent,
)

logger = logging.getLogger("intelligent-interruption-agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

load_dotenv()

# Latency tracking
_latency_tracker = defaultdict(lambda: {"start": None, "events": []})


def log_latency(stage: str, event: str, details: str = "") -> None:
    now = time.time()
    tracker = _latency_tracker[stage]
    
    if tracker["start"] is None:
        tracker["start"] = now
        elapsed = 0.0
    else:
        elapsed = now - tracker["start"]
    
    tracker["events"].append((now, event, details))
    
    elapsed_ms = elapsed * 1000
    print(f"⏱️  [{stage:15s}] {event:30s} | +{elapsed_ms:7.2f}ms | {details}")


class IntelligentInterruptionAgent(Agent):
    """
    A voice agent with intelligent interruption handling.

    Backchannels like "yeah", "ok", "uh-huh" are ignored while the agent
    is speaking. Commands like "stop", "wait", "no" will interrupt immediately.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly. You are a helpful voice assistant. "
                "For stories, give an entertaining story to the user, based on their request. "
                "Keep your tone friendly and conversational. "
                "Do not use emojis, asterisks, markdown, or special characters. "
                "IMPORTANT: If a user pauses you mid-story and later asks to continue or resume, "
                "pick up EXACTLY where you left off in the story. You can see what you already said "
                "in the conversation history - continue from that point, don't start over."
            ),
        )

    async def on_enter(self) -> None:
        # Generate an initial greeting when the agent starts
        log_latency("PIPELINE", "AGENT STARTED", "Initial greeting")
        self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help them today."
        )
    
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        log_latency("LLM", "REQUEST STARTED", f"Processing user message")
        # Use the text_content property which correctly extracts text from the message
        text = new_message.text_content
        if text:
            preview = text[:50] + "..." if len(text) > 50 else text
            log_latency("LLM", "USER MESSAGE", f'"{preview}"')
        else:
            log_latency("LLM", "USER MESSAGE", "(no text content)")

    @function_tool
    async def tell_story(self, context: RunContext, topic: str) -> str:
        """Called when the user asks for a story.

        Args:
            topic: The topic or theme of the story
        """
        logger.info(f"Generating story about: {topic}")
        return f"Tell a very brief 3-4 sentence story about {topic}. Keep it short to avoid rate limits."

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ) -> str:
        """Called when the user asks for weather information.

        Args:
            location: The location they are asking for
            latitude: The latitude of the location
            longitude: The longitude of the location
        """
        logger.info(f"Looking up weather for {location}")
        return f"The weather in {location} is sunny with a temperature of 72 degrees."


def create_interruption_handler(
    session: AgentSession,
    classifier: InterruptionClassifier,
) -> callable:
    """
    Create an event handler for user_input_transcribed events.

    This handler implements the intelligent interruption logic:
    - If agent is speaking and utterance is a backchannel -> do nothing
    - If agent is speaking and utterance is a command/other -> force interrupt
    - If agent is not speaking -> let normal flow handle it
    """

    def on_user_input_transcribed(ev: UserInputTranscribedEvent) -> None:
        transcript = ev.transcript
        is_final = ev.is_final
        
        # DEBUG: Confirm handler is being called
        print(f"[DEBUG] user_input_transcribed event received: is_final={is_final}, transcript='{transcript}'")
        
        # Get current agent state FIRST
        agent_state = session.agent_state
        is_speaking = agent_state == "speaking"
        
        # Log STT event
        stage = "STT"
        if is_final:
            log_latency(stage, "FINAL TRANSCRIPT RECEIVED", f'"{transcript}"')
        else:
            log_latency(stage, "PARTIAL TRANSCRIPT", f'"{transcript}"')

        # Classify the utterance (enable debug for final transcripts to see normalization)
        classify_start = time.time()
        intent = classifier.classify(transcript, debug=is_final)
        classify_time = (time.time() - classify_start) * 1000
        log_latency("CLASSIFIER", "CLASSIFICATION COMPLETE", f"intent={intent.value} ({classify_time:.2f}ms)")

        # VERBOSE: Print all recognized speech prominently
        # Use different formatting when agent is speaking to make it stand out
        final_marker = "[FINAL]" if is_final else "[partial]"
        
        if is_speaking:
            # Highlight user speech during agent speaking - this is important!
            print(f"\n{'!'*60}")
            print(f"🎤🔊 USER SPEAKING OVER AGENT {final_marker}:")
            print(f"   \"{transcript}\"")
            print(f"   Intent: {intent.value} | Action: {'INTERRUPT' if intent != InterruptionIntent.IGNORE_BACKCHANNEL else 'IGNORE (backchannel)'}")
            print(f"{'!'*60}\n")
        else:
            # Check if this backchannel will be ignored
            will_ignore = is_final and intent == InterruptionIntent.IGNORE_BACKCHANNEL
            action = "IGNORE (clearing turn)" if will_ignore else "RESPOND"
            print(f"\n{'='*60}")
            print(f"🎤 USER SPEECH {final_marker}: \"{transcript}\"")
            print(f"   Agent state: {agent_state} | Intent: {intent.value} | Action: {action}")
            print(f"{'='*60}\n")

        # Handle backchannels even when agent is listening
        # We don't want the agent to respond to "okay", "yeah", "uh-huh" etc.
        if not is_speaking:
            if is_final and intent == InterruptionIntent.IGNORE_BACKCHANNEL:
                logger.info(f"IGNORING backchannel while listening: '{transcript}'")
                # Clear the user turn to prevent agent from responding to pure backchannels
                session.clear_user_turn()
                return
            logger.debug("Agent not speaking - normal flow will handle input")
            return

        # Decision based on classification
        if intent == InterruptionIntent.IGNORE_BACKCHANNEL:
            logger.info(f"IGNORING backchannel while speaking: '{transcript}'")
            # Clear the user turn to prevent the system from trying to respond
            if is_final:
                session.clear_user_turn()
            return

        elif intent == InterruptionIntent.INTERRUPT_COMMAND:
            logger.info(f"INTERRUPTING due to command: '{transcript}'")
            # Force interrupt the current speech
            try:
                session.interrupt(force=True)
                logger.info("Successfully interrupted agent speech")
                # Clear the user turn so the command itself doesn't trigger a response
                # (we don't want the agent to respond to "hold on" or "stop")
                if is_final:
                    session.clear_user_turn()
            except RuntimeError as e:
                logger.warning(f"Could not interrupt: {e}")

        elif intent == InterruptionIntent.INTERRUPT_OTHER:
            logger.info(f"INTERRUPTING due to new content: '{transcript}'")
            # Force interrupt the current speech - but DON'T clear the turn
            # because this is legitimate new content the user wants to discuss
            try:
                session.interrupt(force=True)
                logger.info("Successfully interrupted agent speech")
            except RuntimeError as e:
                logger.warning(f"Could not interrupt: {e}")

    return on_user_input_transcribed


# Create the agent server
from livekit.agents import AgentServer

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """Prewarm the VAD model for faster startup."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    """Main entrypoint for the voice agent session."""

    # Set up logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Create classifier with default configuration
    # You can customize the backchannel/command word sets here
    classifier_config = InterruptionClassifierConfig(
        backchannel_words={
            "yeah", "yes", "yep", "yup",
            "ok", "okay",
            "hmm", "hm", "mhm",
            "uh-huh", "uh huh", "uhuh",
            "mm-hmm", "mm hmm", "mmhmm",
            "right", "sure", "alright", "yessir",
            "ah", "oh",
            "i see", "got it", "gotcha", "haan", "accha", "theek hai", "samajh gaya", "samajh gayi"
        },
        command_phrases={
            "stop", "wait", "no", "cancel",
            "hold on", "hold", "pause",
            "quiet", "shut up", "enough",
            "never mind", "nevermind",
            "actually", "but",
            "nahi", "ruko", "bas bas"
        },
        interrupt_on_non_backchannel=True,
    )
    classifier = InterruptionClassifier(classifier_config)

    # Create LLM, STT, and TTS instances
    llm_instance = GeminiLLM(model="gemini-2.0-flash-001")
    stt_instance = deepgram.STT(model="nova-3")
    tts_instance = deepgram.TTS(model="aura-2-andromeda-en")
    
    print(f"\n{'*'*60}")
    print(f"LLM PROVIDER: {llm_instance.provider}")
    print(f"LLM MODEL: {llm_instance.model}")
    print(f"STT PROVIDER: Deepgram")
    print(f"STT MODEL: {stt_instance.model if hasattr(stt_instance, 'model') else 'nova-3'}")
    print(f"TTS PROVIDER: Deepgram")
    print(f"TTS MODEL: {tts_instance.model if hasattr(tts_instance, 'model') else 'aura-2-andromeda-en'}")
    print(f"{'*'*60}\n")
    print("LATENCY TRACKING ENABLED - All events will be logged with timestamps\n")

    # Create the agent session with interruptions DISABLED by default
    # We will handle interruptions manually via the classifier
    #
    # Using:
    # - LLM: Gemini 2.0 Flash (requires GOOGLE_API_KEY)
    # - STT: Deepgram Nova-3 (requires DEEPGRAM_API_KEY)
    # - TTS: Deepgram Aura-2 (requires DEEPGRAM_API_KEY)
    session = AgentSession(
        # Speech-to-text via Deepgram
        stt=stt_instance,
        # LLM via Gemini
        llm=llm_instance,
        # Text-to-speech via Deepgram
        tts=tts_instance,
        # Turn detection
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # CRITICAL: Disable automatic interruptions
        # We handle interruptions manually based on utterance classification
        allow_interruptions=False,
        # CRITICAL: Keep processing audio even when agent is speaking!
        # This allows us to capture and classify user speech during agent output
        discard_audio_if_uninterruptible=False,
        # Other settings
        preemptive_generation=True,
        min_endpointing_delay=0.5,
        max_endpointing_delay=3.0,
    )

    # Register the interruption handler
    interruption_handler = create_interruption_handler(session, classifier)
    session.on("user_input_transcribed", interruption_handler)
    
    # Log when speech is created
    @session.on("speech_created")
    def _on_speech_created(ev: SpeechCreatedEvent) -> None:
        source = ev.source
        log_latency("PIPELINE", f"SPEECH CREATED ({source})", 
                   f"user_initiated={ev.user_initiated}")

    # Log agent state changes for visibility
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent) -> None:
        old_state = ev.old_state
        new_state = ev.new_state
        
        if new_state == "thinking":
            log_latency("LLM", "THINKING STARTED", f"{old_state} -> {new_state}")
        elif new_state == "speaking":
            log_latency("TTS", "SPEECH GENERATION STARTED", f"{old_state} -> {new_state}")
            print(f"\n🤖 AGENT SPEAKING...")
        elif old_state == "speaking" and new_state != "speaking":
            log_latency("TTS", "SPEECH GENERATION COMPLETE", f"{old_state} -> {new_state}")
            print(f"🤖 AGENT STOPPED SPEAKING (now {new_state})")
        elif new_state == "listening":
            log_latency("PIPELINE", "READY FOR INPUT", f"{old_state} -> {new_state}")

    # Log what the agent says
    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if hasattr(item, "role") and item.role == "assistant":
            content = ""
            if hasattr(item, "content"):
                if isinstance(item.content, str):
                    content = item.content
                elif isinstance(item.content, list):
                    for part in item.content:
                        if hasattr(part, "text"):
                            content += part.text
            if content:
                log_latency("LLM", "RESPONSE TEXT GENERATED", f'"{content[:50]}..."')
                print(f"\n{'─'*60}")
                print(f"🤖 AGENT SAID: \"{content}\"")
                print(f"{'─'*60}\n")

    # Set up metrics logging with latency tracking
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        m = ev.metrics
        
        # Log LLM metrics
        if hasattr(m, "llm") and m.llm:
            llm_metrics = m.llm
            ttft_ms = llm_metrics.ttft * 1000 if hasattr(llm_metrics, "ttft") else 0
            tokens_per_sec = llm_metrics.tokens_per_second if hasattr(llm_metrics, "tokens_per_second") else 0
            prompt_tokens = llm_metrics.prompt_tokens if hasattr(llm_metrics, "prompt_tokens") else 0
            completion_tokens = llm_metrics.completion_tokens if hasattr(llm_metrics, "completion_tokens") else 0
            
            log_latency("LLM", "METRICS", 
                f"TTFT={ttft_ms:.1f}ms | {tokens_per_sec:.1f} tok/s | "
                f"{prompt_tokens}+{completion_tokens} tokens")
        
        # Log TTS metrics
        if hasattr(m, "tts") and m.tts:
            tts_metrics = m.tts
            ttfb_ms = tts_metrics.ttfb * 1000 if hasattr(tts_metrics, "ttfb") else 0
            audio_duration = tts_metrics.audio_duration if hasattr(tts_metrics, "audio_duration") else 0
            chars_count = tts_metrics.characters_count if hasattr(tts_metrics, "characters_count") else 0
            
            log_latency("TTS", "METRICS",
                f"TTFB={ttfb_ms:.1f}ms | Audio={audio_duration:.2f}s | "
                f"{chars_count} chars")
        
        # Log STT metrics
        if hasattr(m, "stt") and m.stt:
            stt_metrics = m.stt
            audio_duration = stt_metrics.audio_duration if hasattr(stt_metrics, "audio_duration") else 0
            
            log_latency("STT", "METRICS", f"Processed {audio_duration:.2f}s of audio")
        
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage() -> None:
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
        
        # Print latency summary
        print(f"\n{'='*80}")
        print("📊 LATENCY SUMMARY")
        print(f"{'='*80}")
        for stage, tracker in sorted(_latency_tracker.items()):
            if tracker["events"]:
                events = tracker["events"]
                if len(events) > 1:
                    total_time = (events[-1][0] - events[0][0]) * 1000
                    print(f"\n{stage}:")
                    print(f"  Total events: {len(events)}")
                    print(f"  Total time: {total_time:.2f}ms")
                    print(f"  Events:")
                    for i, (ts, event, details) in enumerate(events):
                        if i == 0:
                            print(f"    {i+1}. {event:40s} | {details}")
                        else:
                            delta = (ts - events[i-1][0]) * 1000
                            print(f"    {i+1}. {event:40s} | +{delta:7.2f}ms | {details}")
                else:
                    print(f"\n{stage}: {events[0][1]} | {events[0][2]}")
        print(f"{'='*80}\n")

    # Faster shutdown - close session properly
    async def shutdown_session() -> None:
        await log_usage()
        await session.aclose()

    ctx.add_shutdown_callback(shutdown_session)

    # Start the session
    await session.start(
        agent=IntelligentInterruptionAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )

    logger.info(
        "Intelligent Interruption Agent started. "
        "Using: Gemini 2.0 Flash (LLM), Deepgram Nova-3 (STT), Deepgram Aura-2 (TTS). "
        "Backchannels will be ignored while speaking. "
        "Commands like 'stop' or 'wait' will interrupt."
    )


if __name__ == "__main__":
    cli.run_app(server)
