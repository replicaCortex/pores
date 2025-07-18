{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell rec {
  buildInputs = with pkgs; [
    python312Packages.uv
  ];

  shellHook = ''
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt

    export LD_LIBRARY_PATH="${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
  '';
}
