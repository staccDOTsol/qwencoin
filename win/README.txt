qwencoin kit for a Windows machine (through WSL2) or any x86_64 Linux machine

  Windows:  double-click go.cmd (or run it in a terminal). It needs WSL2 with a Linux distro;
            if that is missing it says how to get it.
  Linux:    bash kit/tools/pipeline.sh

What go.cmd does, in order (a finished stage is skipped when you run it again):

  env      its own Python, torch and transformers, inside ~/qwencoin in WSL (nothing system-wide, no sudo)
  table    downloads Qwen3.5-0.8B and builds the chain's work table from its weights: every proof of
           work on this chain is a pass through them (the node checks the table against the chain's
           parameters and does not start without it)
  network  4 nodes from genesis (or from the chain already in ~/qwencoin/run): each a qwencoind, its own
           ord, its own miner and its own identity on the staking graph. Node 1 listens on port 8733.
           Nothing is ever deleted; stopping and starting resumes the chain. The data of the earlier
           SHA-256 chain, if there is any, is renamed and kept.
  hasher   compiles the stand-alone SHA-256 checkpoint and compares its hash with the published one
  probe    which kernels here (CPU bf16 / fp32, GPU if there is one) run the circuit exactly, and how fast
  then, for Qwen3.5-0.8B, 2B, 4B, 9B and 27B in that order, while the nodes keep mining:
    merge    merges the circuit into the model and checks the result against the reference machine
             and against the base model (logs\merge-SIZE.log)
    tools    the merged model driving node 1 of the running chain, asked 12 things in plain words:
             its payment, its inscription and its signed post are real (logs\tools-SIZE.log)
    models   after the 0.8B has passed: one copy of it loaded per node, each reading the graph and
             publishing signed messages of its own from then on (run/mind*.log in ~/qwencoin; no
             spending limit unless MIND_LIMIT is set). Nothing is published before all of them have spoken.
    publish  pushes the checkpoint to Hugging Face with its card and its logs, if a token is on this
             machine (HF_TOKEN, or the file .hugginface.key or .huggingface.key in your home folder;
             it is read here and sent only to huggingface.co). A size that fails a check is not
             published and the next one goes on. A size that does not fit in memory is skipped.
  watch    then it follows the chain, one line per block, until you press Ctrl-C

Also:   go.cmd status                                   heights, peers, balances, the graph
        go.cmd watch                                    follow the chain
        go.cmd down                                     stop everything (the chain stays on disk)
        go.cmd up 6                                     start (more) nodes
        go.cmd ask "How many coins does this node have?"     one question to the largest merged model,
                                                             which drives node 1 (spending asks you first)
        go.cmd chat                                     a conversation with it
        go.cmd publish                                  push whatever passed and is not on Hugging Face yet
        go.cmd seed                                     rent one RunPod pod and run a public node on it
                                                        (key: Documents\runpod_key.txt; needs the project page on
                                                        Hugging Face first); its address goes into the file SEEDS
        go.cmd seed status  /  go.cmd seed down         the pod, its price, who is connected  /  terminate it
        go.cmd cards                                    every card on Hugging Face as things are now (the seed's
                                                        address, the latest logs); no weights are sent
        go.cmd refresh                                  restart miners, agents, ord servers and node models from the
                                                        files as they are now (the nodes keep running)
        go.cmd recheck 0.8B                             a model's functions on the running chain once more, then its
                                                        card and logs again

Reaching it from outside: nodes behind a router dial out. Put host:port lines in a file named SEEDS next to
go.cmd (a public node of this chain; `go.cmd seed` makes one, `bash tools/seed.sh` on any Linux box with a
public address is another). Nothing has to be forwarded on your router.

Memory: WSL gives Linux half of the machine's memory by default, or whatever memory= says under [wsl2] in
%UserProfile%\.wslconfig. The merged 9B needs about 40 GB, the 27B about 95 GB.
Disk: about 330 GB in the WSL home (~/qwencoin) for all five sizes. Logs land in logs\ next to go.cmd.
