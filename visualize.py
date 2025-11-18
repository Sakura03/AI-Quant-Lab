from nofx.visualization import visualize_snapshots


if __name__ == "__main__":
    visualize_snapshots("results/default_20251118_221840/snapshots",
                        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT"],
                        save_path="results/default_20251118_221840/viz/balance.png")
