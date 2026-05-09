# Release notes instructions

## Categories
- **Features** - new user-facing functionality, commands, analysis modes, or outputs
- **Fixes** - bug fixes, correctness improvements, and reproducibility fixes
- **Performance** - faster or more memory-efficient computations
- **Documentation** - documentation, examples, tutorials, and citation metadata
- **Dependencies** - dependency, packaging, CI, and compatibility changes that affect users

## Skip
Do not generate release-note entries for:
- CI-only changes with no user-visible package effect
- test-only changes
- internal refactors with no user-visible behavior change

## Style
- Write for users of the package, not only for maintainers
- Mention affected modules, command-line interfaces, and output files when possible
- Use concise present-tense bullets
- Prefer concrete behavior changes over implementation details
