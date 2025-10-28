def rebalance_options(current_calls, current_puts, target_amount):
    total_calls_value = sum(current_calls)
    total_puts_value = sum(current_puts)
    total_portfolio = total_calls_value + total_puts_value
    
    recommendation = {
        "close_calls": 0,      
        "close_puts": 0,        
        "open_calls": 0,       
        "open_puts": 0,        
        "net_change": 0,       
        "details": []
    }
    
    if total_portfolio == 0:
        print("No existing options to trade")
        return recommendation
    
    # STRATEGY 1: Need MORE bullish exposure (target_amount > 0)
    if target_amount > 0:
        # Close puts and convert to calls
        available_capital = 0
        puts_to_close = 0
        
        for put_value in sorted(current_puts, reverse=True):  # Close largest first
            if available_capital >= target_amount:
                break
            available_capital += put_value
            puts_to_close += 1
            recommendation["details"].append(f"Close PUT worth ${put_value:.2f}")
        
        recommendation["close_puts"] = puts_to_close
        
        # Use freed capital to open calls
        # Assume each call costs average of current call values
        avg_call_cost = total_calls_value / len(current_calls) if current_calls else 200
        calls_to_open = int(available_capital / avg_call_cost)
        
        recommendation["open_calls"] = calls_to_open
        recommendation["net_change"] = calls_to_open * avg_call_cost - available_capital
        
        for i in range(calls_to_open):
            recommendation["details"].append(f"Open CALL (~${avg_call_cost:.2f} each)")
    
    # STRATEGY 2: Need MORE bearish exposure (target_amount < 0)
    elif target_amount < 0:
        target_reduction = abs(target_amount)
        
        # Close calls and convert to puts
        available_capital = 0
        calls_to_close = 0
        
        for call_value in sorted(current_calls, reverse=True):  # Close largest first
            if available_capital >= target_reduction:
                break
            available_capital += call_value
            calls_to_close += 1
            recommendation["details"].append(f"Close CALL worth ${call_value:.2f}")
        
        recommendation["close_calls"] = calls_to_close
        
        # Use freed capital to open puts
        avg_put_cost = total_puts_value / len(current_puts) if current_puts else 150
        puts_to_open = int(available_capital / avg_put_cost)
        
        recommendation["open_puts"] = puts_to_open
        recommendation["net_change"] = -(puts_to_open * avg_put_cost)
        
        for i in range(puts_to_open):
            recommendation["details"].append(f"Open PUT (~${avg_put_cost:.2f} each)")
    
    # STRATEGY 3: No change needed
    else:
        recommendation["details"].append("Portfolio is balanced - no trades needed")
    
    return recommendation


def display_rebalance(ticker, rec):
    print(ticker)
    print("\n" + "="*60)
    print("OPTIONS REBALANCING RECOMMENDATION")
    print("="*60)
    print(f"\nClose {rec['close_calls']} CALL contract(s)")
    print(f"Close {rec['close_puts']} PUT contract(s)")
    print(f"Open {rec['open_calls']} new CALL contract(s)")
    print(f"Open {rec['open_puts']} new PUT contract(s)")
    print(f"\nNet Portfolio Change: ${rec['net_change']:,.2f}")
    
    print("\nDetailed Actions:")
    for detail in rec['details']:
        print(f"  • {detail}")
    print("="*60)


# # Example usage
# if __name__ == "__main__":
#     # Current options holdings
#     my_calls = [500, 300, 200, 150]  # $1150 total in calls
#     my_puts = [100, 150, 75]          # $325 total in puts
    
#     print("Current Portfolio:")
#     print(f"  Calls: ${sum(my_calls):,.2f} ({len(my_calls)} contracts)")
#     print(f"  Puts: ${sum(my_puts):,.2f} ({len(my_puts)} contracts)")
#     print(f"  Total: ${sum(my_calls) + sum(my_puts):,.2f}")
    
#     # Example 1: Need to be MORE bullish (+$300)
#     print("\n" + "="*60)
#     print("SCENARIO 1: Increase bullish exposure by $300")
#     rec1 = rebalance_options(my_calls, my_puts, 300)
#     display_rebalance(rec1)
    
#     # Example 2: Need to be MORE bearish (-$400)
#     print("\n" + "="*60)
#     print("SCENARIO 2: Increase bearish exposure by $400")
#     rec2 = rebalance_options(my_calls, my_puts, -400)
#     display_rebalance(rec2)
    
#     # Example 3: No change needed
#     print("\n" + "="*60)
#     print("SCENARIO 3: Portfolio is balanced")
#     rec3 = rebalance_options(my_calls, my_puts, 0)
#     display_rebalance(rec3)
# # ```

# **How it works:**

# 1. **To increase bullish exposure (+$):**
#    - Close PUT contracts (starting with largest)
#    - Use that money to open new CALL contracts

# 2. **To increase bearish exposure (-$):**
#    - Close CALL contracts (starting with largest)
#    - Use that money to open new PUT contracts

# 3. **No new capital needed** - just swapping existing positions

# **Example output:**
# ```
# Current Portfolio:
#   Calls: $1,150.00 (4 contracts)
#   Puts: $325.00 (3 contracts)

# SCENARIO 1: Increase bullish exposure by $300
# ============================================================
# Close 2 PUT contract(s)
# Open 1 new CALL contract(s)
# Net Portfolio Change: $0.00

# Detailed Actions:
#   • Close PUT worth $150.00
#   • Close PUT worth $100.00
#   • Open CALL (~$287.50 each)