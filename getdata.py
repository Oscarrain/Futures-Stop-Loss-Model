import dai

df = dai.query("""select * from cn_future_bar1m where instrument = 'ec2412.INE'""", filter={"date":["2024-07-29 09:00:00", "2024-07-29 15:00:00"]}).df()

# output df into csv
df.to_csv('data/ec2412_2024-07-29.csv', index=False)