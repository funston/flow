"""Per-command operations — one implementation per BACH command.

Every surface (CLI shell, ACP for Zed, aria HTTP endpoints for
VS Code, Emacs over comint) calls into these functions. Surfaces
are thin adapters that translate the function's callbacks into
their UI primitives. No surface contains its own copy of a
command's logic.
"""
