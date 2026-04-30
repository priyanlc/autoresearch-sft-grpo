# Friction Log

This file records non-trivial failures encountered during autoresearch runs. The intent is to make it easy for the human author to reconstruct *what broke, what was tried, and what stuck* without re-deriving it from raw logs — both for the next session and for the follow-up blog post on the autoresearch pattern.

A "non-trivial failure" is anything that took more than a one-line config tweak to resolve (or is still unresolved). Examples that qualify:

- `qutlass` failed to build during `pip install`
- The `__bases__` patch ran but PEFT still couldn't see `FPQuantLinear` modules
- GRPO crashed with a TRL tensor-mismatch error
- The model loaded but generation produced empty output
- `transformers_modules` cache poisoning after a version bump

Examples that do *not* need an entry: a typo in `train.py`, an obvious off-by-one in a reward function, a missing import.

---

## Template

Copy this block for each new entry. **Newer entries go at the top of the Entries section below.**

```markdown
### F-NNN — <one-line summary>

- **timestamp:** <YYYY-MM-DD HH:MM UTC>
- **phase:** <env_install | model_load | sft | grpo | eval | cleanup | other>
- **signature:**

  ```
  <actual error message or observed symptom; truncate stacktrace to ~10 most informative lines>
  ```

- **hypothesized root cause:** <best guess; say "uncertain" if uncertain>
- **attempts:**
  - <attempt 1> → <worked | no effect | made it worse | crashed differently>
  - <attempt 2> → <outcome>
  - ...
- **final state:** <resolved | worked-around | punted | open>
- **notes:** <optional — context that doesn't fit above; e.g., "happens only with `BATCH_SIZE > 1`">
```

---

## Entries

<!-- Append new entries below this line, newest first. Use sequential ids: F-001, F-002, ... -->
