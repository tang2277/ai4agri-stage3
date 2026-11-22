(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  // --- Chart 1: NDVI 分类分布（周口 2024 小麦季） ---
  var chart1 = echarts.init(document.getElementById('chart-ndvi-dist'), null, { renderer: 'svg' });
  chart1.setOption({
    tooltip: { trigger: 'item', appendToBody: true },
    legend: {
      top: 'bottom',
      textStyle: { color: ink, fontSize: 12 }
    },
    color: ['#c44d4d', '#c07a2d', '#d4b34a', '#8fbc5e', '#2d7a4f'],
    series: [{
      type: 'pie',
      radius: ['40%', '70%'],
      avoidLabelOverlap: true,
      itemStyle: {
        borderColor: '#fff',
        borderWidth: 2
      },
      label: {
        formatter: '{b}\n{d}%',
        color: ink,
        fontSize: 12
      },
      animation: false,
      data: [
        { value: 0.1, name: '< 0 水体/裸土' },
        { value: 30.1, name: '0-0.2 稀疏植被' },
        { value: 36.3, name: '0.2-0.4 中等植被' },
        { value: 33.5, name: '0.4-0.6 茂盛作物' },
        { value: 0.0, name: '> 0.6 密林' }
      ]
    }]
  });
  window.addEventListener('resize', function() { chart1.resize(); });

  // --- Chart 2: 数据源对比 ---
  var chart2 = echarts.init(document.getElementById('chart-datasource'), null, { renderer: 'svg' });
  chart2.setOption({
    tooltip: { appendToBody: true },
    legend: { top: 'bottom', textStyle: { color: ink, fontSize: 12 } },
    grid: { left: 60, right: 30, top: 30, bottom: 80 },
    xAxis: {
      type: 'category',
      data: ['零注册', '国内速度', 'COG支持', 'API稳定', '免费'],
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted, fontSize: 11 }
    },
    yAxis: {
      type: 'value',
      name: '评分 (0-5)',
      nameTextStyle: { color: muted },
      max: 5,
      axisLine: { show: false },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } },
      axisLabel: { color: muted, fontSize: 11 }
    },
    series: [
      {
        name: 'Planetary Computer ✅',
        type: 'bar',
        data: [5, 4.5, 5, 4.5, 5],
        itemStyle: { color: accent, borderRadius: [4, 4, 0, 0] },
        barWidth: 22,
        animation: false
      },
      {
        name: 'Copernicus Dataspace',
        type: 'bar',
        data: [2, 2, 5, 2, 5],
        itemStyle: { color: accent2, borderRadius: [4, 4, 0, 0] },
        barWidth: 22,
        animation: false
      },
      {
        name: 'Google Earth Engine',
        type: 'bar',
        data: [1, 0, 5, 5, 5],
        itemStyle: { color: '#c07a2d', borderRadius: [4, 4, 0, 0] },
        barWidth: 22,
        animation: false
      }
    ]
  });
  window.addEventListener('resize', function() { chart2.resize(); });

})();
