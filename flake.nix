{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
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
      in rec {
        inherit overlay overlays;
        devShell = pkgs.mkShell rec {
          name = "venv";
          venvDir = "./.venv";
          buildInputs = let
            systemPackages = with pkgs; [
              pre-commit

              gfortran
              # ^ required by scipy
            ];
            pythonPackages = with pkgs.python310Packages; [
              python
              ipython
              venvShellHook

              poetry

              jax
              (jaxlib-bin.override { cudaSupport = true; })
            ];
          in pythonPackages ++ systemPackages;

          postVenvCreation = postShellHook + ''
            poetry install -E haiku -E flax -E logging
            pre-commit install
          '';

          postShellHook =''
            export OPENBLAS=${pkgs.openblas}/lib
            # ^ required by scipy
            export LD_LIBRARY_PATH+=:${pkgs.zlib}/lib
            # ^ required by scipy
            export LD_LIBRARY_PATH+=:${pkgs.gcc-unwrapped.lib}/lib
            # ^ required by scipy
          '' ;
        };
      });
}
