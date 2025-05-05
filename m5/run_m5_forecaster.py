#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run M5 forecaster with command line arguments
"""

import argparse
from m5_forecaster import M5Forecaster

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='M5 Forecaster')
    parser.add_argument('--level', type=int, choices=[13, 14, 15], default=13,
                       help='Which level to process (13=HOBBIES, 14=HOUSEHOLD, 15=FOODS)')
    parser.add_argument('--data-path', type=str, default='data/m5-forecasting-accuracy',
                       help='Path to M5 data directory')
    parser.add_argument('--super-speed', action='store_true', default=True,
                       help='Use super speed mode (default)')
    parser.add_argument('--speed', action='store_true',
                       help='Use regular speed mode (slightly slower than super speed)')
    
    args = parser.parse_args()
    
    # If --speed is provided, turn off super-speed
    if args.speed:
        args.super_speed = False
        
    return args

def main():
    """Main function"""
    args = parse_args()
    
    print(f"Training M5 forecaster for level {args.level}")
    print(f"Using {'SUPER_SPEED' if args.super_speed else 'SPEED'} mode")
    
    forecaster = M5Forecaster(
        data_path=args.data_path,
        level=args.level,
        super_speed=args.super_speed
    )
    
    forecaster.run()

if __name__ == "__main__":
    main()
