{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = github:numtide/flake-utils;
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system overlays;
          config = { allowUnfree = true; };
        };
        overlay = (final: prev: { });
        overlays = [ overlay ];
      in
      rec {
        inherit overlay overlays;
        devShell = pkgs.mkShell
          rec {
            name = "venv";
            venvDir = "./.venv";
            buildInputs = with pkgs; [
              python310Packages.python
              python310Packages.ipython
              python310Packages.venvShellHook

              python310Packages.poetry

              python310Packages.jax
              (python310Packages.jaxlib-bin.override { cudaSupport = true; })
            ];

            postVenvCreation = ''
              poetry install
            '';

            postShellHook = ''
            '';
          };
      });
}
