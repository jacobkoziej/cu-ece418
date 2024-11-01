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

        python = pkgs.python3;

        python-pkgs = python.withPackages (
          python-pkgs: with python-pkgs; [
            ffmpeg-python
            ipython
            numpy
          ]
        );

      in
      {
        devShells.default = pkgs.mkShell {
          packages =
            [
              python-pkgs
            ]
            ++ (with pkgs; [
              black
              ffmpeg
              ruff
              treefmt2
            ]);
        };

        formatter = pkgs.nixfmt-rfc-style;
      }
    );
}
