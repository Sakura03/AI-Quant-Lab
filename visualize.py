from nofx.utils import visualize_snapshots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="可视化快照文件")
    parser.add_argument("--snapshot-path", type=str, help="快照文件目录")
    parser.add_argument("--symbols", nargs="+", help="交易对列表")
    parser.add_argument("--save-path", type=str, help="图片的保存目录")
    args = parser.parse_args()

    visualize_snapshots(args.snapshot_path, symbols=args.symbols, save_path=args.save_path)
