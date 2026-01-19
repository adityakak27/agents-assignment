"""
Utterance classifier for intelligent interruption handling.

This module provides a lightweight, configurable classifier that distinguishes:
- Backchannels (passive acknowledgements like "yeah", "ok", "uh-huh")
- Commands (explicit interruption intents like "stop", "wait", "no")
- Other content (genuine new input that should interrupt)

The classifier is used to determine whether user speech during agent output
should be ignored (backchannels) or cause an interruption (commands/other).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Set


class InterruptionIntent(Enum):
    """Classification result for an utterance."""

    IGNORE_BACKCHANNEL = "ignore_backchannel"
    """Utterance is purely a backchannel - should be ignored while agent is speaking."""

    INTERRUPT_COMMAND = "interrupt_command"
    """Utterance contains an explicit interruption command."""

    INTERRUPT_OTHER = "interrupt_other"
    """Utterance contains non-backchannel content - treat as genuine interruption."""


@dataclass
class InterruptionClassifierConfig:
    """Configuration for the interruption classifier."""

    # Single-word backchannels (passive acknowledgements)
    backchannel_words: Set[str] = field(
        default_factory=lambda: {
            "yeah",
            "yes",
            "yep",
            "yup",
            "ok",
            "okay",
            "hmm",
            "hm",
            "mhm",
            "uh-huh",
            "uh huh",
            "uhuh",
            "mm-hmm",
            "mm hmm",
            "mmhmm",
            "right",
            "sure",
            "ah",
            "oh",
            "i see",
            "got it",
            "gotcha",
            "accha",
            "theek hai",
            "samajh gaya",
            "samajh gayi"
        }
    )

    # Explicit interruption commands
    command_phrases: Set[str] = field(
        default_factory=lambda: {
            "stop",
            "wait",
            "no",
            "cancel",
            "hold on",
            "hold",
            "pause",
            "quiet",
            "shut up",
            "enough",
            "never mind",
            "nevermind",
            "actually",
            "but",
            "nahi",
            "ruko",
        }
    )

    # Whether to treat any non-backchannel content as an interruption
    # If False, only explicit commands will interrupt
    interrupt_on_non_backchannel: bool = True


class InterruptionClassifier:
    """
    Classifies user utterances to determine interruption intent.

    The classifier normalizes input text and checks:
    1. If any command phrase is present -> INTERRUPT_COMMAND
    2. If the entire utterance is backchannels only -> IGNORE_BACKCHANNEL
    3. Otherwise -> INTERRUPT_OTHER (if configured) or IGNORE_BACKCHANNEL
    """

    def __init__(self, config: InterruptionClassifierConfig | None = None):
        self.config = config or InterruptionClassifierConfig()
        # Pre-compile patterns for efficiency
        self._setup_patterns()

    def _setup_patterns(self) -> None:
        """Compile regex patterns for matching."""
        # Normalize all phrases to lowercase
        self._backchannels = {self._normalize(b) for b in self.config.backchannel_words}
        self._commands = {self._normalize(c) for c in self.config.command_phrases}

        # Build command detection pattern (word boundary matching)
        escaped_commands = [re.escape(cmd) for cmd in sorted(self._commands, key=len, reverse=True)]
        if escaped_commands:
            self._command_pattern = re.compile(
                r"\b(" + "|".join(escaped_commands) + r")\b", re.IGNORECASE
            )
        else:
            self._command_pattern = None

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove punctuation except hyphens (for "uh-huh")
        text = re.sub(r"[^\w\s-]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _is_backchannel_only(self, normalized_text: str) -> bool:
        """Check if the entire utterance consists only of backchannels."""
        if not normalized_text:
            return True  # Empty text is treated as backchannel

        # Check if entire text matches a single backchannel phrase
        if normalized_text in self._backchannels:
            return True

        # Try splitting into words and checking if all are backchannels
        words = normalized_text.split()
        if not words:
            return True

        # Check single words
        if len(words) == 1:
            return words[0] in self._backchannels

        # For multi-word utterances, check if it's a known multi-word backchannel
        # or if all individual words are backchannels
        if normalized_text in self._backchannels:
            return True

        # Check all individual words
        for word in words:
            if word not in self._backchannels:
                # Check if this word plus next forms a multi-word backchannel
                found_as_part_of_phrase = False
                for bc in self._backchannels:
                    if " " in bc and word in bc.split():
                        # This word is part of a multi-word backchannel
                        # Check if the full phrase is in the text
                        if bc in normalized_text:
                            found_as_part_of_phrase = True
                            break
                if not found_as_part_of_phrase:
                    return False

        return True

    def classify(self, transcript: str, debug: bool = False) -> InterruptionIntent:
        """
        Classify an utterance to determine interruption intent.

        Args:
            transcript: The user's speech transcript (interim or final)
            debug: If True, print debug info about classification

        Returns:
            InterruptionIntent indicating how the agent should handle this utterance
        """
        if not transcript:
            return InterruptionIntent.IGNORE_BACKCHANNEL

        normalized = self._normalize(transcript)

        if debug:
            print(f"[CLASSIFIER DEBUG] Original: '{transcript}' -> Normalized: '{normalized}'")
            print(f"[CLASSIFIER DEBUG] Is in backchannels: {normalized in self._backchannels}")

        if not normalized:
            return InterruptionIntent.IGNORE_BACKCHANNEL

        # Check for command phrases first (highest priority)
        if self._command_pattern and self._command_pattern.search(normalized):
            if debug:
                print(f"[CLASSIFIER DEBUG] Matched command pattern")
            return InterruptionIntent.INTERRUPT_COMMAND

        # Check if it's purely backchannels
        if self._is_backchannel_only(normalized):
            if debug:
                print(f"[CLASSIFIER DEBUG] Is backchannel only")
            return InterruptionIntent.IGNORE_BACKCHANNEL

        # Non-backchannel content
        if debug:
            print(f"[CLASSIFIER DEBUG] Not a backchannel, treating as interrupt_other")
        if self.config.interrupt_on_non_backchannel:
            return InterruptionIntent.INTERRUPT_OTHER
        else:
            return InterruptionIntent.IGNORE_BACKCHANNEL

    def should_interrupt(self, transcript: str) -> bool:
        """
        Convenience method to check if an utterance should cause interruption.

        Args:
            transcript: The user's speech transcript

        Returns:
            True if the agent should interrupt, False if it should continue
        """
        intent = self.classify(transcript)
        return intent in (InterruptionIntent.INTERRUPT_COMMAND, InterruptionIntent.INTERRUPT_OTHER)


# Default classifier instance with standard configuration
default_classifier = InterruptionClassifier()
