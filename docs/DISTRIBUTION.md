# bhc Distribution Channels

Tracking all places where bhc is or could be available.

## Live

| Channel | Install Command | Link |
|---------|-----------------|------|
| GitHub Releases | `curl -fsSL https://bhc.raskell.io/install.sh \| sh` | [releases](https://github.com/raskell-io/bhc/releases) |
| Homebrew | `brew install raskell-io/tap/bhc` | [homebrew-tap](https://github.com/raskell-io/homebrew-tap) |

## Planned

| Channel | Priority | Notes |
|---------|----------|-------|
| Cargo | Ready | `cargo install bhc` - publishes on stable release (requires LLVM 18) |
| Nix Flake | Medium | `nix run github:raskell-io/bhc` |
| Scoop | Low | Windows support needed first |

## Not Yet (requires more platform support)

| Channel | Blocker |
|---------|---------|
| asdf | Need stable release + more platforms |
| aqua | Need stable release |
| mise | Uses asdf plugin |
| AUR | Need Linux aarch64 support |
| Nixpkgs | Need stable release |
| WinGet | Need Windows builds |
| Chocolatey | Need Windows builds |
| Docker/GHCR | Could add for CI/CD use cases |

## Current Platform Support

| Platform | Status |
|----------|--------|
| Linux x86_64 | ✅ Available |
| macOS aarch64 (Apple Silicon) | ✅ Available |
| macOS x86_64 (Intel) | ❌ Runner retired |
| Linux aarch64 | ❌ Cross-compilation issues |
| Windows | ❌ Not yet supported |

## Plugin/Package repos

| Repo | Purpose |
|------|---------|
| [raskell-io/homebrew-tap](https://github.com/raskell-io/homebrew-tap) | Homebrew formula (shared with hx) |
