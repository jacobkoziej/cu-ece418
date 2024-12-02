{
  description = "The Cooper Union - ECE 418: Digital Video Processing";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs =
    inputs:

    inputs.flake-utils.lib.eachDefaultSystem (
      system:

      let
        pkgs = import inputs.nixpkgs {
          inherit system;
        };

        lib = pkgs.lib;

        python = pkgs.python3;

        python-pkgs = python.withPackages (
          python-pkgs: with python-pkgs; [
            einops
            ffmpeg-python
            ipython
            msgpack
            numpy
            scipy
          ]
        );

      in
      {
        devShells.default = pkgs.mkShell (
          let
            pre-commit-bin = "${lib.getBin pkgs.pre-commit}/bin/pre-commit";
          in
          {
            packages =
              [
                python-pkgs
              ]
              ++ (with pkgs; [
                black
                ffmpeg
                mdformat
                pre-commit
                ruff
                shfmt
                toml-sort
                treefmt2
                yamlfmt
              ]);

            shellHook = ''
              ${pre-commit-bin} install --allow-missing-config > /dev/null
            '';
          }
        );

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
